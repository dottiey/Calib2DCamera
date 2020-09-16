import cv2
import pickle
import numpy as np
from calib import CVRodrigues
from calib import undis_points
from grid_warping_3 import load_warp_data
import matplotlib.pyplot as plt
from pathlib import Path
from ransac import *
from mpl_toolkits import mplot3d
import imageio
import os
import matplotlib.cm as cm

def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

def createMatrixList(model):
    HinvList = []
    TransList = []
    params = load_warp_data(model)
    it = 0
    for r in params[4]:
        # print(cv2.Rodrigues(r)[0])
        RM = cv2.Rodrigues(r)[0]
        # print("Number: ", it)
        # print("rmatrrix", RM)
        # print("tvec", params[3][it])
        tvec = np.array(params[3][it]).reshape(len(params[3][it]), 1)
        r1r2t = np.append(RM[:,0:2], tvec, axis = 1)
        r1r2t = r1r2t/r1r2t[2,2]
        # print("r1r2t", r1r2t)
        # print("   ")
        H = params[1].dot(r1r2t)
        Hinv = np.linalg.inv(H)

        r1 = RM[:,0]
        r2 = RM[:,1]
        r3 = np.cross(r1, r2)

        r1r2r3t = np.hstack((r1.reshape(3,1), r2.reshape(3,1), r3.reshape(3,1), tvec))
        Transf = np.append(r1r2r3t, np.array((0,0,0,1)).reshape(1,4), axis = 0)
        TransList.append(Transf)
        HinvList.append(Hinv)
        it = it + 1
    return (TransList,HinvList)

def distance_pt_plane(a, b, c, d, PT):
    distance=(np.abs(a*PT[0]+b*PT[1]+c*PT[2]+d))/np.sqrt(a*a+b*b+c*c)
    return distance

def plot_plane(a, b, c, d):
    xx, yy = np.mgrid[-20:100, -20:100]
    return xx, yy, (-d - a * xx - b * yy) / c

def COG(img, width, height, filterLevel):
    COGArrayX = []
    COGArrayY = []
    columnPos = np.arange(0, height)
    # img=cv2.imread(pathImg, 0)
    img16=img.astype(np.int16)
    for col in range (0, width):
        img16[:,col] = np.maximum((img16[:,col]-filterLevel), np.zeros(height))
        columnValue = np.sum(img16[:, col])
        sumCol = np.maximum(columnValue, 0)
        sumValue = columnPos*img16[:,col]
        # print(sumCol)
        if(sumCol!=0):
            cogvalue = np.sum(sumValue)/sumCol
            COGArrayX.append(col)
            COGArrayY.append(cogvalue)
    return (COGArrayX, COGArrayY)

def process_2D_to_3D(u, v, Homography, params):
    u = np.asarray(u)
    v = np.asarray(v)
    u = u.reshape(len(u),1)
    v = v.reshape(len(v),1)

    pts_uv2=np.append(u,v,axis=1)
    pointUndis = undis_points(pts_uv2, params[1], params[0])
    uni = np.zeros(pointUndis.shape[0])+1
    uni = uni.reshape(pointUndis.shape[0],1)
    uv = pointUndis.reshape(pointUndis.shape[0],2)
    uv1 = np.append(uv,uni,axis=1)

    xyz = Homography.dot(uv1.T)
    XX = xyz[0]/xyz[2]
    YY = xyz[1]/xyz[2]
    #
    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(u, v,'.')
    # axs[0].grid(True)
    # axs[1].plot(uv[:,0], uv[:,1],'.')
    plt.show()
    return (XX, YY)

def rototransl(X, Y, Transf, TransfRifWorld):
    TransfRifWorldInv = np.linalg.inv(TransfRifWorld)
    TT = TransfRifWorldInv.dot(Transf)
    zero = np.zeros(len(X))
    uni = zero + 1
    zero = zero.reshape(zero.shape[0],1)
    uni = uni.reshape(uni.shape[0],1)
    X = X.reshape(X.shape[0],1)
    Y = Y.reshape(Y.shape[0],1)
    xyz1 = np.append(X, np.append(Y,(np.append(zero , uni, axis=1)), axis=1), axis=1)
    XYZnew=TT.dot(xyz1.T)

    return(XYZnew[0], XYZnew[1], XYZnew[2])

def posetoplane(a, b, c, d, XYZ):
    Nx = a/d
    Ny = b/d
    Nz = c/d

    meanX=np.mean(XYZ[0])
    meanY=np.mean(XYZ[1])
    meanZ=np.mean(XYZ[2])
    # 3.14159/180
    a1 = np.cos(-np.arctan(Nx/Nz))
    a2 = np.sin(-np.arctan(Nx/Nz))

    a3 = np.cos(np.arctan(Ny/Nz))
    a4 = np.sin(np.arctan(Ny/Nz))

    hb1 = np.append(a1,np.append(0,np.append(a2,0)))
    hb2 = np.append(0,np.append(1,np.append(0,0)))
    hb3 = np.append(-a2,np.append(0,np.append(a1,0)))
    hb4 = np.append(0,np.append(0,np.append(0,1)))
    Hbeta = np.append(hb1.reshape(4,1),np.append(hb2.reshape(4,1),np.append(hb3.reshape(4,1),hb4.reshape(4,1),axis=1),axis=1),axis=1).T

    ha1 = np.append(1,np.append(0,np.append(0,0)))
    ha2 = np.append(0,np.append(a3,np.append(-a4,0)))
    ha3 = np.append(0,np.append(a4,np.append(a3,0)))
    ha4 = np.append(0,np.append(0,np.append(0,1)))
    Halfa = np.append(ha1.reshape(4,1),np.append(ha2.reshape(4,1),np.append(ha3.reshape(4,1),ha4.reshape(4,1),axis=1),axis=1),axis=1).T

    c0 = np.array([1,0,0,0])
    c1 = np.array([0,1,0,0])
    c2 = np.array([0,0,1,0])
    c3 = np.append(-meanX,np.append(-meanY,np.append(-meanZ,1)))
    T = np.append(c0.reshape(4,1),np.append(c1.reshape(4,1),np.append(c2.reshape(4,1),c3.reshape(4,1),axis=1),axis=1),axis=1)

    Hplane = Hbeta.dot(Halfa.dot(T))
    return Hplane

def calibraPianoLaser(model,pathImgLaser, numPianoRiferimento, thresholdCOG):
    # model:"guido"
    print(model, pathImgLaser, numPianoRiferimento, thresholdCOG)
    params = load_warp_data(model)
    # print("params:", params)
    (TransList,HinvList) = createMatrixList(model)
    # pathImgLaser = "opencv\\laser\\*.tif"
    path = Path(pathImgLaser)
    imgs = [cv2.imread(str(i), 0) for i in path.parent.glob(path.name)]
    itr = 0
    Xcloud = np.array([])
    Ycloud = np.array([])
    Zcloud = np.array([])
    base = numPianoRiferimento
    for img in imgs:

        (t, s1) = COG(img, img.shape[1], img.shape[0], thresholdCOG)
        print("iterazione ", itr, "img ", itr+base )

        (XX, YY) = process_2D_to_3D(t, s1, HinvList[itr+base], params)
        (newX, newY, newZ) = rototransl(XX, YY, TransList[itr+base], TransList[base])
        itr = itr + 1
        # plt.plot(XX, YY,'.')
        # plt.show()
        newX = newX.reshape(newX.shape[0],1)
        newY = newY.reshape(newY.shape[0],1)
        newZ = newZ.reshape(newZ.shape[0],1)

        Xcloud = np.append(Xcloud,newX)
        Ycloud = np.append(Ycloud,newY)
        Zcloud = np.append(Zcloud,newZ)


    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)

    max_iterations = 10000
    goal_inliers = Xcloud.shape[0] * 0.3

    XYZcloud = np.array([Xcloud, Ycloud, Zcloud])
    # ax.scatter3D(XYZcloud[0], XYZcloud[1], XYZcloud[2])

    m, b = run_ransac(XYZcloud.T, estimate, lambda x, y: is_inlier(x, y, 0.005), 3, goal_inliers, max_iterations)
    a, b, c, d = m
    dddd=distance_pt_plane(a, b, c, d, XYZcloud)
    col=dddd*255/max(dddd)

    ax.scatter3D(XYZcloud[0], XYZcloud[1], XYZcloud[2], c=col, s=20, cmap='rainbow')

    xx, yy, zz = plot_plane(a, b, c, d)
    print("plane coefficients: ",a," ",b," ",c," ",d)
    ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.2))
    plt.show()

    Hplane = posetoplane(a, b, c, d, XYZcloud)
    Hplaneinv = np.linalg.inv(Hplane)
    H = TransList[base].dot(Hplaneinv)

    r1r2t = np.append(H[0:3:,0:2], H[0:3,3].reshape(3,1), axis = 1)
    H2 = params[1].dot(r1r2t)
    Homography_camera_laser = np.linalg.inv(H2)
    print("Homography_camera_laser: ", Homography_camera_laser)
    (X_LUT, Y_LUT) = generateLUT(imgs[0].shape[0],imgs[0].shape[1], params, Homography_camera_laser)

    pathLaser = os.path.dirname(os.path.dirname(pathImgLaser))
    filename = "LUT"
    pathLUT = os.path.join(pathLaser, filename)
    print("pathLUT:", pathLUT)
    if not os.path.exists(pathLUT):
        os.mkdir(pathLUT)

    save_LUT_png(X_LUT, Y_LUT, pathLUT)

    testLUT(X_LUT, Y_LUT, pathImgLaser)
    return Homography_camera_laser

def generateLUT(width, height, params, Hom):
    X_LUT = np.full((width, height),1.0)
    Y_LUT = np.full((width, height),1.0)
    v = np.arange(0, height)
    for col in range (0, width):
        u = np.full((height, 1), col)
        (XX, YY) = process_2D_to_3D(u, v, Hom, params)
        X_LUT[col,:] = XX
        Y_LUT[col,:] = YY
    return (X_LUT, Y_LUT)

def save_LUT_png(X_LUT, Y_LUT, path):
    pathLUTX = os.path.join(path, 'LUTX.png')
    pathLUTY = os.path.join(path, 'LUTY.png')

    X_LUT1=X_LUT-X_LUT.min()
    X_LUT2=X_LUT1*100
    X_LUT3=X_LUT2.astype('uint16')

    Y_LUT1=Y_LUT-Y_LUT.min()
    Y_LUT2=Y_LUT1*100
    Y_LUT3=Y_LUT2.astype('uint16')

    imageio.imwrite(pathLUTX, X_LUT3)
    imageio.imwrite(pathLUTY, Y_LUT3)
    return

def testLUT(X_LUT, Y_LUT, pathImgLaser):
    path = Path(pathImgLaser)
    imgs = [cv2.imread(str(i), 0) for i in path.parent.glob(path.name)]
    itr = 0
    Xcloud = np.array([])
    Ycloud = np.array([])
    for img in imgs:
        (t, s1) = COG(img, img.shape[1], img.shape[0], 80)

        s1Int = [int(i) for i in s1]
        Xcloud = np.append(Xcloud,X_LUT[s1Int, t])
        Ycloud = np.append(Ycloud,Y_LUT[s1Int, t])

    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    ax.scatter3D(Xcloud, Ycloud, np.zeros(Xcloud.size), s=20, cmap='rainbow')
    plt.show()

    return



def main():
    import argparse
    from pathlib import Path
    from textwrap import dedent

    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='command')  # Save command in variable command
    sp.required = True  # A command is mandatory

# calibraPianoLaser("guido", pathImgLaser, numPianoRiferimento, thresholdCOG)

    sub1 = sp.add_parser('calib_laser_plane')
    sub1.add_argument('model', help='Calibration model to save')
    sub1.add_argument('images', help='Image file of the lines laser')
    sub1.add_argument('rif', type=int,  help='number of image to use as reference of coordinates')
    sub1.add_argument('thresh', type=int,  help='threshold for extract laser line with COG algorithm')
    # sub1.add_argument('-t', '--text', action='store_true', help='Save output model as text')

    args = ap.parse_args()

    path = Path(args.images)
    print("pathImages  ",path)

    if not path.parent.is_dir():
        print('ERROR!', path.parent, 'is not a directory')
        return

    if args.command == 'calib_laser_plane':
        print('Starting calibration of laser plane...')
        print('args.model', args.model)
        print('args.images', args.images)
        print('args.thresh', args.thresh)
        print('args.rif', args.rif)
        print('Expanding dir', path.parent, 'glob', path.name)
        calibraPianoLaser(args.model, path, args.rif, args.thresh)


if __name__ == '__main__':
    main()

# thresholdCOG = 80
# numPianoRiferimento = 20
# pathImgLaser = "opencv\\laser\\*.tif"
# calibraPianoLaser("guido", pathImgLaser, numPianoRiferimento, thresholdCOG)
