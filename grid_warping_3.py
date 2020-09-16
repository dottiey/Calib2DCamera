import cv2
import pickle
import numpy as np
from pathlib import Path
from itertools import product
from matplotlib import pyplot

def bigshow(img, name='image', size=None):
    """Display the image in a window, safely."""
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if size is not None:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    else:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    if size is not None:
        cv2.resizeWindow(name, size[0], size[1])
    else:
        cv2.resizeWindow(name, 500, 500)

    # If the window is closed while waitKey(0) mainloop will hang
    # therefore we loop waiting for a key for a while, then
    # checking if window was closed
    while True:
        k = cv2.waitKey(500)
        prop = cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
        if k != -1 or prop < 1:  # Key pressed or window closed
            break
    cv2.destroyAllWindows()

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 3)
    return img

def init_calib(grid_size):
    nos = grid_size[0] * grid_size[1]
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    camera_points = []
    world_points = []

def show_img(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name,img)
    cv2.waitKey(200)

def SendImg(array):
	array = []
	array= array.append(30)
	return array

def verify_axis(pathImg, model, grid_size):
    params = load_warp_data(model)
    dist, mtx, camera_matrix, tvecs, rvecs = params
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:grid_size[0],0:grid_size[1]].T.reshape(-1,2)
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    path=Path(pathImg)
    imgs = [cv2.imread(str(i), 0) for i in path.parent.glob(path.name)]
    for i, img in enumerate(imgs):
        ret, corners = cv2.findCirclesGrid(img, grid_size, None)
        if ret == True:
            corners2 = cv2.cornerSubPix(img,corners,(11,11),(-1,-1),criteria)
            # Find the rotation and translation vectors.
            ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            img = draw(img,corners2,imgpts)
            cv2.imshow('img',img)
            k = cv2.waitKey(10) & 0xFF
    cv2.destroyAllWindows()
    return


def detect_warp(imgs, grid_size, chess_circles, wheelbase_mm):
    # Number of squares
    nos = grid_size[0] * grid_size[1]
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    camera_points = []
    world_points = []
    for i, img in enumerate(imgs):
    # for img in imgs:
        # Find corners
        if(chess_circles):
            ret, corners = cv2.findChessboardCorners(img, grid_size, None)
            corners=cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        else:
            ret, corners = cv2.findCirclesGrid(img, grid_size, None)
            # print("corners: ", corners)
            # import pdb; pdb.set_trace()
        if not ret:
            print('pattern was not found on number ', i)
            # Skip this image if pattern was not found
            continue
        # Get better corner positions

        cv2.drawChessboardCorners(img, (grid_size[0], grid_size[1]), corners, ret)
        show_img("chess",img)

        # Get position of centermost corner
        cent_corn = corners[corners.shape[0] // 2]
        print('Center corner is', cent_corn)

        # Points in 2D camera space
        src = np.array(corners).reshape((nos, 2))

        # Points in 3D world space
        obj_points = np.zeros((nos, 3), dtype=np.float32)
        obj_points[:, :2] = np.indices(grid_size).T.reshape(-1, 2)*wheelbase_mm

        # import pdb; pdb.set_trace()
        camera_points.append(src)
        world_points.append(obj_points)
        # H = cv2.findHomography(obj_points, src)
        # import pdb; pdb.set_trace()
    h, w = imgs[0].shape
    # Pass a single image
    rms, mtx, d_coef, rvecs, tvecs = cv2.calibrateCamera(world_points, camera_points, (w, h), None, None)

    cam_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, d_coef, (w, h), 1, (w, h))

    print('RMS', rms)
    print('mtx', mtx)
    print('camera matrix', cam_mtx)
    print('distorsion coefficients', d_coef)

    return (mtx, d_coef, cam_mtx, rvecs, tvecs)


def fix_warp(img, params):
    mtx, d_coef, cam_mtx = params
    return cv2.undistort(img, mtx, d_coef, None, cam_mtx)


def save_warp_data(path, params, save_text=False):
    if save_text:
        for i, p in enumerate(params):
            np.savetxt(path + '_{}.txt'.format(i), p)
    with open(path, 'wb') as outfile:
        pickle.dump(params, outfile)


def load_warp_data(path):
    with open(path, 'rb') as outfile:
        return pickle.load(outfile)


def main():
    import argparse
    from pathlib import Path
    from textwrap import dedent

    description = dedent("""\
        Tool for calibrating camera using a series of grid pattern images
        as well as undistortimg images. See example below.""")

    example = dedent("""\
        # Example usage:

        ## First step

        Generate model for a 5x4 grid
          This command will save model both as pkl (mymodel.pkl) and text
          (mymodel_0.txt, mymodel_1.txt, mymodel_2.txt, with flag -t)

            grid_warping.py generate -s -t -W 5 -H 4 mymodel /path/to/images/\*.png

        ## Second step
        Apply model to some images
          The pkl model is used in this step

            grid_warping.py apply mymodel.pkl /dist/images\*.png ./out/dir/""")

    ap = argparse.ArgumentParser(
            description=description,
            epilog=example,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='command')  # Save command in variable command
    sp.required = True  # A command is mandatory

    sub1 = sp.add_parser('generate')
    sub1.add_argument('model', help='Calibration model to save')
    sub1.add_argument('images', help='Image file of the grid to scan')
    sub1.add_argument('-W', '--width', type=int, default=9, help='Grid width')
    sub1.add_argument('-H', '--height', type=int, default=7, help='Grid height')
    sub1.add_argument('-s', '--show', action='store_true', help='Show results')
    sub1.add_argument('-t', '--text', action='store_true', help='Save output model as text')

    sub2 = sp.add_parser('apply')
    sub2.add_argument('model', help='Calibration model to apply')
    sub2.add_argument('images', help='Image files to fix')
    sub2.add_argument('outdir', nargs='?', default=None, help='Path of folder where to save images')

    args = ap.parse_args()

    path = Path(args.images)
    if not path.parent.is_dir():
        print('ERROR!', path.parent, 'is not a directory')
        return

    if args.command == 'generate':
        print('Generating warp grid...')
        print('Expanding dir', path.parent, 'glob', path.name)
        imgs = [cv2.imread(str(i), 0) for i in path.parent.glob(path.name)]
        params = detect_warp(imgs, (args.width, args.height))
        save_warp_data(args.model, params, args.text)
        if args.show:
            print('Showing results on reference image')
            for img in imgs:
                fixed = fix_warp(img, params)
                bigshow(fixed)
    elif args.command == 'apply':
        print('Applying warp model...')
        params = load_warp_data(args.model)
        for image in path.parent.glob(path.name):
            print('Carico', image)
            name = Path(image).parts[-1]
            img = cv2.imread(image, 0)
            fixed = fix_warp(img, params)
            if args.outdir is not None:
                outp = str(args.outdir / name)
                print('Writing to file', outp)
                cv2.imwrite(outp, fixed)
            else:
                print('Displaying')
                bigshow(fixed)

if __name__ == '__main__':
    main()
