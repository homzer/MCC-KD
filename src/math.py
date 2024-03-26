import multiprocessing
from math import isclose
from tokenize import TokenError
from typing import Union

from sympy import simplify, N
from sympy.parsing.latex import parse_latex, LaTeXParsingError
from sympy.parsing.sympy_parser import parse_expr


def is_digit(s):
    try:
        float(str(s).replace(",", ""))
        return True
    except ValueError:
        return False


def symbolic_equal(a: str, b: str):
    def _parse(s):
        for f in [parse_latex, parse_expr]:
            try:
                return f(s)
            except LaTeXParsingError:
                pass
            except SyntaxError:
                pass
            except TokenError:
                pass
            except TypeError:
                pass
            except AttributeError:
                pass
            except ValueError:
                pass
        return s

    a = _parse(a)
    b = _parse(b)
    if isinstance(a, str) or isinstance(b, str):
        return False  # Parsing failed

    try:
        if simplify(a - b) == 0:
            return True
    except TypeError:
        pass
    except AttributeError:
        pass

    try:
        if isclose(N(a), N(b), rel_tol=1e-3):
            return True
    except TypeError:
        pass
    except AttributeError:
        pass

    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=1, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()


def math_equal(
        prediction: Union[bool, float, str],
        reference: Union[float, str],
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    if is_digit(prediction) and is_digit(reference):
        prediction = float(str(prediction).replace(",", ""))
        reference = float(str(reference).replace(",", ""))
        for item in [reference / 100, reference, reference * 100]:
            if isclose(item, prediction, rel_tol=1e-4):
                return True
        return False

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    # deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or \
            (prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ['{', "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str == ref_str:
        return True

    # [a, b] vs. [c, d], return a==c and b==d
    if (prediction.startswith("[") and prediction.endswith("]")) and \
            (reference.startswith("[") and reference.endswith("]")) or \
            (prediction.startswith("(") and prediction.endswith(")")) and \
            (reference.startswith("(") and reference.endswith(")")):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all([math_equal(pred_parts[i], ref_parts[i]) for i in range(len(pred_parts))]):
                return True

    # symbolic equal with sympy
    # if call_with_timeout(symbolic_equal_process, prediction, reference):
    #     return True
    if symbolic_equal(prediction, reference):
        return True

    return False


def _test_math_equal():
    print(math_equal("0.0833333333333333", "\\frac{1}{12}"))
    print(math_equal("1.5s", "(3/2)s"))
    print(symbolic_equal("\\frac{x}{7}+\\frac{2}{7}", "\\frac{x+2}{7}"))
    print(math_equal("\\frac{x}{7}+\\frac{2}{7}", "\\frac{x+2}{7}"))


if __name__ == "__main__":
    _test_math_equal()
