import argparse

from control import PGM
from problem import Problem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="python file with model description")
    parser.add_argument("output", help="directory to save outputs")
    parser.add_argument("path_mp4", help="directory with set of png images produced by pgm to create mp4 video")
    args = parser.parse_args()

    if args.path_mp4:
        create_video(args.path_mp4)


    problem = Problem(default_settings_filename='defaults.py')
    problem.load_model(args.model)

    control = PGM(problem, figname = args.output)
    control.run()

def create_video(path_to_images):
    import os
    os.system(f'cd {path_to_images}')
    os.system('ffmpeg -framerate 25 -pattern_type glob -i "*.png"  -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4')
    os.system('cd -')

if __name__ == '__main__':
    main()
