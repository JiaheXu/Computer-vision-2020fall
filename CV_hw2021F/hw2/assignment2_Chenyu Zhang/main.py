import PointPicker
import ProjectiveDistortion
import DrawPolygon
import ProjectiveRectification
import OpenCVRectification


def main():
    # Pick the points in the images
    ph = PointPicker.PointPicker()
    # DrawPolygon.DrawPolygon(ph)

    # Distort the points
    ph_distort = ProjectiveDistortion.ProjectiveDistortion(ph)

    # Display the distorted shape
    DrawPolygon.DrawPolygon(ph_distort)

    # Calculate and display the rectified shape
    Hp = ProjectiveRectification.ProjectiveRectification(ph_distort)
    print('\nHomography compute by me:\n', Hp)

    # Call OpenCVRectification to estimate the homography
    H = OpenCVRectification.OpenCVRectification(ph_distort, ph)
    print('\nHomography compute by OpenCV:\n', H)
    
if __name__=="__main__":
    main()
