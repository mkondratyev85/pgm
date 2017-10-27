import argparse

from control import PGM, load_model, load_settings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="python file with model description")
    parser.add_argument("output", help="directory to save outputs")

    args = parser.parse_args()

    #fname = 'benchmarks/push.py'
    #model_prop = load_model(fname)

    # figname = '/tmp/t7/'
    # control = PGM(model_prop, figname = figname)

    model_prop = load_model(args.model)
    control = PGM(model_prop, figname = args.output)

    control.init_(model_prop)
    sec = lambda t: t * 365.25 * 24 * 3600 * 10**6 # Converts Myrs back to seconds
    control.run_(MaxT=sec(10000000000))

if __name__ == '__main__':
    main()
