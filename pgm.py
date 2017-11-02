import argparse

from control import PGM
from problem import Problem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="python file with model description")
    parser.add_argument("output", help="directory to save outputs")
    args = parser.parse_args()

    problem = Problem(default_settings_filename='defaults.py')
    problem.load_model(args.model)

    control = PGM(problem, figname = args.output)
    control.run()

if __name__ == '__main__':
    main()
