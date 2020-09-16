import cv2
from pathlib import Path
from grid_warping_3 import detect_warp
from grid_warping_3 import show_img
from grid_warping_3 import save_warp_data
from grid_warping_3 import load_warp_data
from grid_warping_3 import verify_axis
import numpy as np

arrayImg=[]

def calibSquare(path_img, grid_W, grid_H, distance):
    path=Path(path_img)
    imgs = [cv2.imread(str(i), 0) for i in path.parent.glob(path.name)]
    print (type(imgs[0]))
    params = detect_warp(imgs, (grid_W, grid_H), True, distance)
    imgAr = np.array(imgs[0])
    print("num immagini: ",len(imgs))
    # print("params: ",params)
    array=imgAr.tolist()
    d_coef = params[1][0]
    mtx = params[0]
    camera_matrix = params[2]
    tvecs = np.array(params[4]).reshape(len(params[4]),3)
    rvecs = np.array(params[3]).reshape(len(params[3]),3)
    return (d_coef,mtx,camera_matrix,tvecs,rvecs)

def calibCircles(path_img, grid_W, grid_H, distance):
    path=Path(path_img)
    imgs = [cv2.imread(str(i), 0) for i in path.parent.glob(path.name)]
    # params = detect_warp(imgs, (7, 7), False, 3.75)
    params = detect_warp(imgs, (grid_W, grid_H), False, distance)
    imgAr = np.array(imgs[0])
    print("num immagini: ",len(imgs))
    # print("params: ",params)
    array=imgAr.tolist()
    #return  d_coef,mtx,cam_mtx,tvecs,rvecs
    d_coef = params[1][0]
    mtx = params[0]
    camera_matrix = params[2]
    tvecs = np.array(params[4]).reshape(len(params[4]),3)
    rvecs = np.array(params[3]).reshape(len(params[3]),3)
    return (d_coef, mtx, camera_matrix, tvecs, rvecs)

def findCorners(array,grid_shape1, grid_shape2):
    img=np.asarray(array,dtype=np.uint8)
    ret, corners = cv2.findChessboardCorners(img, (grid_shape1, grid_shape2), None)
    if ret==True:
        cv2.drawChessboardCorners(img, (grid_shape1, grid_shape2), corners, ret)
    show_img("ff",img)
    return ret


def findCirclesGrid(array,grid_shape1, grid_shape2):
    img=np.asarray(array,dtype=np.uint8)
    ret, centers = cv2.findCirclesGrid(img, (grid_shape1, grid_shape2), None)
    if ret==True:
        cv2.drawChessboardCorners(img, (grid_shape1, grid_shape2), centers, ret)
    show_img("ff",img)
    return ret

def projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs):
    objectPoints2=np.asarray(objectPoints,dtype=np.float64)
    cameraMatrix2=np.asarray(cameraMatrix,dtype=np.float64)
    distCoeffs2=np.asarray(distCoeffs,dtype=np.float64)
    rvec2=np.asarray(rvec,dtype=np.float64)
    tvec2=np.asarray(tvec,dtype=np.float64)
    # points3D = np.float32([[0.5 , 0.5 , 0]])
    ptsRepr = cv2.projectPoints(objectPoints2, rvec2, tvec2, cameraMatrix2, distCoeffs2)
    ttc=np.array(ptsRepr[0]).reshape(len(ptsRepr[0]),2)
    return ttc

def undis_points(pts_uv, camera_matrix, dist_coefs):
    # pts_uv = np.array([[1.0, 1.0], [1.0, 2.0]])
    camera_matrix2=np.asarray(camera_matrix,dtype=np.float64)
    dist_coefs2=np.asarray(dist_coefs,dtype=np.float64)
    pts_uv2=np.asarray(pts_uv,dtype=np.float64)
    ptsUnd = cv2.undistortPoints(pts_uv2.reshape(len(pts_uv2),1,2), camera_matrix2, dist_coefs2, None, camera_matrix2)
    ttc=np.array(ptsUnd[:]).reshape(len(ptsUnd[:]),2)
    return ttc

def mapUndistort(pts_uv, camera_matrix, dist_coefs, width, height):
    pts_uv2=np.asarray(pts_uv,dtype=np.uint32)
    camera_matrix2=np.asarray(camera_matrix,dtype=np.float64)
    dist_coefs2=np.asarray(dist_coefs,dtype=np.float64)
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix2, dist_coefs2, None, camera_matrix2, (width, height), cv2.CV_32FC1)

    # xcor = mapx[pts_uv2[:,0],pts_uv2[:,1]]
    # ycor = mapy[pts_uv2[:,0],pts_uv2[:,1]]
    return(mapx,mapy)

def tieni(array):
    img=np.asarray(array,dtype=np.uint8)
    arrayImg.append(img)
    return len(arrayImg)

def calibra(grid_shape1, grid_shape2, wheelbase_mm):
    params=detect_warp(arrayImg, (grid_shape1,grid_shape2), True, wheelbase_mm)
    # mtx, d_coef, cam_mtx = params
    return (params[1][0],params[0],params[2], np.array(params[3]).reshape(len(params[3]),3), np.array(params[4]).reshape(len(params[4]),3))

def testCluster():
    par=mymain()
    return (par[1][0],par[0])

def undistImg(imgArray, d_coef, mtx, cam_mtx):
    img=np.asarray(imgArray,dtype=np.uint8)
    # mtx, d_coef, cam_mtx = params
    d_coef2=np.asarray(d_coef,dtype=np.float64)
    mtx2=np.asarray(mtx,dtype=np.float64)
    cam_mtx2=np.asarray(cam_mtx,dtype=np.float64)
    imgUndis = cv2.undistort(img, mtx2, d_coef2, None, cam_mtx2)
    # show_img("undistImg",imgUndis)
    return np.array(imgUndis)

def GetHomography(objectPointsPlanar, imagePoints):
    objectPointsPlanar2=np.asarray(objectPointsPlanar,dtype=np.float64)
    imagePoints2=np.asarray(imagePoints,dtype=np.float64)
    H = cv2.findHomography(objectPointsPlanar2, imagePoints2)
    return H[0]

def CVRodrigues(rvec):
    rvec2=np.asarray(rvec,dtype=np.float64)
    return cv2.Rodrigues(rvec2)[0]


def main():
    import argparse
    from pathlib import Path
    from textwrap import dedent

    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='command')  # Save command in variable command
    sp.required = True  # A command is mandatory

    sub1 = sp.add_parser('calib_circle')
    sub1.add_argument('model', help='Calibration model to save')
    sub1.add_argument('images', help='Image file of the grid to scan')
    sub1.add_argument('-W', '--width', type=int, default=7, help='Grid width')
    sub1.add_argument('-H', '--height', type=int, default=7, help='Grid height')
    sub1.add_argument('-d', '--distance', type=float, default=1.0, help='wheelbase circles')
    sub1.add_argument('-t', '--text', action='store_true', help='Save output model as text')

    sub2 = sp.add_parser('calib_chessboard')
    sub2.add_argument('model', help='Calibration model to save')
    sub2.add_argument('images', help='Image file of the grid to scan')
    sub2.add_argument('-W', '--width', type=int, default=7, help='Grid width')
    sub2.add_argument('-H', '--height', type=int, default=7, help='Grid height')
    sub2.add_argument('-d', '--distance', type=float, default=1.0, help='square size')
    sub2.add_argument('-t', '--text', action='store_true', help='Save output model as text')

    args = ap.parse_args()
    path = Path(args.images)

    if not path.parent.is_dir():
        print('ERROR!', path.parent, 'is not a directory')
        return

    if args.command == 'calib_circle':
        print('Starting calibration with circle pattern...')
        print('Expanding dir', path.parent, 'glob', path.name)
        print('Pattern with circle', args.width, 'x', args.height, 'distance', args.distance)
        params = calibCircles(path, args.width, args.height, args.distance)
        save_warp_data(args.model, params, args.text)
        verify_axis(path , args.model, (args.width, args.height))

    elif args.command == 'calib_chessboard':
        print('Starting calibration with chessboard...')
        print('Expanding dir', path.parent, 'glob', path.name)
        params = calibSquare(path, args.width, args.height, args.distance)
        save_warp_data(args.model, params, args.text)

if __name__ == '__main__':
    main()

