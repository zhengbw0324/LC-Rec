import transformers
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s", type=str, default="", help="source path of models")
    parser.add_argument("--target", "-t", type=str, default="", help="target path of models")

    args, _ = parser.parse_known_args()

    assert os.path.exists(args.source)
    assert args.target != ""

    model = transformers.AutoModelForCausalLM.from_pretrained(args.source)
    model.save_pretrained(args.target, state_dict=model.state_dict())