import argparse

from control import PGM, load_model
from problem import Problem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="python file with model description")
    parser.add_argument("output", help="directory to save outputs")

    args = parser.parse_args()

    problem = Problem()
    problem.load_settings_from_file(args.model)
    problem.load_model(args.model)

    print(problem['m_eta'])
    model_prop = load_model(args.model)
    print(model_prop['m_eta'])

    control = PGM(problem, figname = args.output)
    control.run()

    # model_prop = load_model(args.model)
    # control = PGM(model_prop, figname = args.output)
    # control.run()

if __name__ == '__main__':
    main()
