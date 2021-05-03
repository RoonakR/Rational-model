from prettytable import PrettyTable
import numpy as np

class Colored(object):

    """Keep it simple, only use `red` and `green` color."""

    RED = '\033[91m'
    GREEN = '\033[92m'

    #: no color
    RESET = '\033[0m'

    def color_str(self, color, s):
        return '{}{}{}'.format(
            getattr(self, color),
            s,
            self.RESET
        )

    def red(self, s):
        return self.color_str('RED', s)

    def green(self, s):
        return self.color_str('GREEN', s)


def print_res(results, combinations, show_list=['auroc', 'auprc'], filename='output.txt', precision=3):
    assert all(x in ['acc', 'prec0', 'prec1', 'rec0', 'rec1', 'auroc', 'auprc', 'minpse'] for x in show_list)
    exceptions = ['CW', 'Loss','AU','Nor','use_lstm', 'callback_decay', 'callback_spar']
    # 'Rational', 'Attention', 'Residual','Hidden',
    colored = Colored()
    title = []
    for k in combinations[0].keys():
        if k not in exceptions:
            title.append(k)
    title += show_list
    t = PrettyTable(title)

    max_values = []
    for idx,name in enumerate(show_list):
        max_values.append(0.0)
        for i in range(len(results)):
            if max_values[idx] < results[i][name]:
                max_values[idx] = results[i][name]

    print(round(max_values[0],precision))
    for idx in range(len(results)):
        value = []
        for ele in combinations[idx].keys():
            if ele not in exceptions:
                value.append(combinations[idx][ele])
        for name_idx,name in enumerate(show_list):
            value.append(colored.red(round(results[idx][name],precision)) 
                if round(results[idx][name],precision) == round(max_values[name_idx],precision) else round(results[idx][name],precision))
        t.add_row(value)
    print(t.get_string())
    with open(filename,'w') as file:
        file.write(t.get_string())