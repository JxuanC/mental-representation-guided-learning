import wget
import pandas as pd
#from retrying import retry
import argparse

def get_kamitaniData_categories(train_csv_dir = './dataset/imageID_training.csv',
                                    test_csv_dir = './dataset/imageID_test.csv'):
    
    train_csv = pd.read_csv(train_csv_dir, header = None)
    test_csv = pd.read_csv(test_csv_dir, header = None)
    train_categories = []
    test_categories = []
    for file in list(train_csv[1]):
        train_categories.append(file.split('_')[0])

    for file in list(test_csv[1]):
        test_categories.append(file.split('_')[0])
    
    return sorted(set(train_categories)), sorted(set(test_categories))

#@retry()
def wget_imageNet(url, save_name, save_path = './dataset/'):
    wget.download(url, out = f"{save_path}/{save_name}.tar")
    print(f"\ndownloaded from {url}")


def download_imagenet(categories, url = "https://image-net.org/data/winter21_whole/", save_path = './dataset/'):
    for id in categories:
        target_url = f"{url}/{id}.tar"
        wget_imageNet(target_url, id, save_path)


def main(args):
    train_categories, test_categories = get_kamitaniData_categories()
    download_imagenet(train_categories, save_path=args.save_path)
    download_imagenet(test_categories, save_path=args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download imagenet')
    parser.add_argument("--save_path", type=str, default="./dataset/", help="save path")
    args = parser.parse_args()

    main(args)