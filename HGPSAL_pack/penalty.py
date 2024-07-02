import numpy as np

def ObjEval(Problem, x, *args):
    try:
        ObjValue = Problem['ObjFunction'](x, *args)
        if 'Stats' in Problem:
            Problem['Stats']['ObjFunCounter'] += 1
    except Exception as e:
        raise RuntimeError(f"User supplied objective function failed with the following error:\n{str(e)}")
    return ObjValue, Problem


def ConsEval(Problem, x, *args):
    try:
        c, ceq = Problem['Constraints'](x, *args)
    except Exception as e:
        raise RuntimeError(f"User supplied function constraints failed with the following error:\n{str(e)}")
    return c, ceq


def penalty2(Problem, x, alg):
    fx, Problem = ObjEval(Problem, x)
    c, ceq = ConsEval(Problem, x)

    term1 = np.sum(alg['lambda'] * np.array(ceq))
    term2 = np.sum(np.array(ceq) ** 2)
    term3 = np.sum(np.maximum(0, alg['ldelta'] + np.array(c) / alg['miu']) ** 2 - alg['ldelta'] ** 2)
    la = fx + term1 + term2 / (2 * alg['miu']) + alg['miu'] * term3 / 2

    return {'fx': fx, 'c': c, 'ceq': ceq, 'la': la}
