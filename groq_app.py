import argparse
import yaml
import os


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml.dev")
parser.add_argument("--mode", type=str, default="cli")
args = parser.parse_args()

if __name__ != "__main__":
    args.config = "config.yaml"

config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

if not config["dev"]:
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
else:
    GROQ_API_KEY = config.get("GROQ", {}).get("key")


