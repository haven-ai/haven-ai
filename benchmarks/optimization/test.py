import argparse 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True)

    args = parser.parse_args()

    print('hello %s' % args.name)
