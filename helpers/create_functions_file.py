file = '../praktikum_tools/functions/gauss.py'

functions = []

def generate_gauss(n, background = None):
    global functions
    parameters = 'x'

    if background:
        functions.append(f'gauss{n}_{background}')
    else:
        functions.append(f'gauss{n}')

    if background == 'linear':
        parameters += ', m, n'
    elif background == 'quadratic':
        parameters += ', a, b, c'
    for i in range(n):
        parameters += f', A{i}'
    for i in range(n):
        parameters += f', sigma{i}'
    for i in range(n):
        parameters += f', mu{i}'
    if background == 'linear':
        return f"def gauss{n}_linear({parameters}): return sum_of_gaussians_linear({parameters})\n"
    if background == 'quadratic':
        return f"def gauss{n}_quadratic({parameters}): return sum_of_gaussians_quadratic({parameters})\n"
    return f"def gauss{n}({parameters}): return sum_of_gaussians({parameters})\n"

def generate_getter():
    func_dict = '{' + ','.join([f"'{function}': {function}" for function in functions]) + '}'
    return f'def getter(n, background = None):\n\treturn {func_dict}' + "[f'gauss{n}' + ('_' + background if background else '')]"


header = '''from tools import exp

def gauss(x, A, sigma, mu): return A * exp(-.5 * ((x - mu) / sigma)**2)

def sum_of_gaussians(x, *args):
    n = len(args) // 3
    As = args[:n]
    sigmas = args[n:2*n]
    mus = args[2*n:3*n]
    return sum([gauss(x, A, sigma, mu) for A, sigma, mu in zip(As, sigmas, mus)])

def linear(x, m, n):
    return m*x + n

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def sum_of_gaussians_linear(x, m, n, *args):
    return linear(x, m, n) + sum_of_gaussians(x, *args)

def sum_of_gaussians_quadratic(x, a, b, c, *args):
    return quadratic(x, a, b, c) + sum_of_gaussians(x, *args)

'''

with open(file, 'w') as f:
    f.write(header)

    f.writelines([generate_gauss(i) for i in range(110)])
    f.write('\n')
    f.writelines([generate_gauss(i, background = 'linear') for i in range(11)])
    f.write('\n')
    f.writelines([generate_gauss(i, background = 'quadratic') for i in range(11)])
    f.write('\n\n')

    f.write(generate_getter())
