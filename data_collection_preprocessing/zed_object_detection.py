########################################################################
#
# Copyright (c) 2021, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    This sample shows how to detect objects and draw 3D bounding boxes around them
    in an OpenGL window

    run it outside the virtual env using python3
"""
import sys
# import ogl_viewer.viewer as gl
import pyzed.sl as sl
from datetime import datetime
import cv2
import os

if __name__ == "__main__":
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode    
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_fps = 15                          # Set fps at 15
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        print("Using SVO file: {0}".format(filepath))
        init_params.set_from_svo_file(filepath)

    # Set runtime parameters
    runtime_parameters = sl.RuntimeParameters()

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable object detection module
    obj_param = sl.ObjectDetectionParameters()
    # Defines if the object detection will track objects across images flow.
    obj_param.enable_tracking = True       # if True, enable positional tracking

    if obj_param.enable_tracking:
        tracking_parameters = sl.PositionalTrackingParameters()
        # tracking_parameters.set_as_static=True
        # tracking_parameters.enable_imu_fusion = False
        tracking_parameters.enable_area_memory= False
        print(tracking_parameters)
        zed.enable_positional_tracking(tracking_parameters)
        
    zed.enable_object_detection(obj_param)

    camera_info = zed.get_camera_information()
    # Create OpenGL viewer
    # viewer = gl.GLViewer()
    # viewer.init(camera_info.calibration_parameters.left_cam)

    # Configure object detection runtime parameters
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 50
    obj_runtime_param.object_class_filter = [sl.OBJECT_CLASS.PERSON]    # Only detect Persons

    # Create ZED objects filled in the main loop
    objects = sl.Objects()
    image = sl.Mat()
    nb_frames = zed.get_svo_number_of_frames()

    # sequence_names = sorted([d for d in os.listdir('/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/') if '2020122' in d and os.path.isdir(os.path.join('/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/',d))])[:-1]
    # sequence_names = sorted([d for d in os.listdir('/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/') if '20210907' in d and os.path.isdir(os.path.join('/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/',d))])[:]
    # # print(sequence_names)
    # sequence_name = [x for x in sequence_names if filepath.split("_")[-1][:-4].replace('-', '') in x][0]

    project_dir = '/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/'
    svo_path = sys.argv[1].replace(project_dir, '')
    
    sequence_name = '%s_%s' % (''.join(svo_path.split('_')[:3]),  svo_path.replace('.svo', '').split("_")[-1].replace('-', ''))
    rgb_dir = os.path.join(project_dir, sequence_name, 'RGB/')
    print(rgb_dir)

    output_txt_name = 'zedBox_' + sequence_name + '.txt'
    out_file = open(os.path.join(project_dir, sequence_name, output_txt_name), 'w')



    while True:
        # Grab an image, a RuntimeParameters object must be given to grab()
        # zed_pose = sl.Pose()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()

            if svo_position >= (nb_frames - 1):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break

            # if svo_position % 10 != 0: continue # sample every 10 frames
            if svo_position % 3 != 0: continue # sample every 3 frames

            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()

            ts = int(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()) / 1000000000
            # print(ts)
            # convert the timestamp to Datetime format
            frame_name = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')
            print(frame_name)

            ### opencv visualization
            # frame = cv2.imread(os.path.join(rgb_dir, frame_name+'.png'))
            
            

            # Retrieve objects
            zed.retrieve_objects(objects, obj_runtime_param)

            # state = zed.get_position(zed_pose, sl.REFERENCE_FRAME.FRAME_WORLD)
            # print(state)

            for obj in objects.object_list:
                zed_bboxes = obj.bounding_box_2d
                state = obj.tracking_state
                if state == sl.OBJECT_TRACKING_STATE.OK:
                    frame = cv2.rectangle(frame, (int(zed_bboxes[0, 0]), int(zed_bboxes[0, 1])), (int(zed_bboxes[2, 0]), int(zed_bboxes[2, 1])), color=(0, 255, 255)) 
                    frame = cv2.putText(frame, str(obj.id), (int((zed_bboxes[0, 0]+zed_bboxes[2, 0])/2), int((zed_bboxes[0, 1] + zed_bboxes[2, 1])/2)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 255), thickness=2)
                    frame = cv2.putText(frame, str(obj.confidence)[:5], (int(zed_bboxes[0, 0]), int(zed_bboxes[0, 1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 255), thickness=1)

                    ### write bbox info to txt file
                    bbox_x = zed_bboxes[0, 0]
                    bbox_y = zed_bboxes[0, 1]
                    bbox_w = zed_bboxes[1, 0] - zed_bboxes[0, 0]
                    bbox_h = zed_bboxes[3, 1] - zed_bboxes[0, 1]
                    line_to_write = [svo_position + 1, obj.id, bbox_x, bbox_y, bbox_w, bbox_h]
                    out_file.write(','.join(map(str, line_to_write))+'\n')


            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)

    
    out_file.close()
            

    # viewer.exit()

    image.free(memory_type=sl.MEM.CPU)
    # Disable modules and close camera
    zed.disable_object_detection()
    zed.disable_positional_tracking()

    zed.close()
