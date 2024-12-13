from fire import Fire
from evaluator import Evaluate
import files2rouge


def main(
    pred_path,
    ref_path,
):
    print(pred_path)

    pred_str = open(pred_path).read().splitlines()
    ref_str = open(ref_path).read().splitlines()
    calculator = Evaluate()
    eval_dict = calculator.evaluate(pred_str, ref_str)
    print(eval_dict)

    files2rouge.run(pred_path, ref_path)

Fire(main)
