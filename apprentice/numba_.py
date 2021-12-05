
try:
#    import numba
    from numba import jit
    from numba import njit
    from numba.typed import List
except ImportError:
    msg="""
    Numba is not available.
    If present, we use its JIT compiler to accelerate computations.
    Try pip install numba --user. You will also require a suitable version of llvmlite.
    """
    print(msg)

    # Dummy functions to not break code if numba is not available
    # https://stackoverflow.com/questions/57774497/how-do-i-make-a-dummy-do-nothing-jit-decorator

    def jit(fastmath=False,parallel=False,forceobj=False):
        def decorator(func):
            return func
        return decorator

    def njit(fastmath=False,parallel=False,forceobj=False):
        def decorator(func):
            return func
        return decorator

    List = list

