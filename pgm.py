import argparse

from control import PGM, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="python file with model description")
    parser.add_argument("output", help="directory to save outputs")

    args = parser.parse_args()

    model_prop = load_model(args.model)
    control = PGM(model_prop, figname = args.output)
    control.run()

if __name__ == '__main__':
    main()
