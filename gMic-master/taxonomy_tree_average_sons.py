import re
import networkx as nx

"""
every bacteria is an object to easily store it's information
"""
class Bacteria:
    def __init__(self, string, val):
        string = string.replace(" ", "")
        lst = re.split("; |__|;", string)
        self.val = val
        # removing letters and blank spaces
        for i in range(0, len(lst)):
            if len(lst[i]) < 2:
                lst[i] = 0
        lst = [value for value in lst if value != 0]
        # Default fall value
        if len(lst) == 0:
            lst = ["Bacteria"]
        self.lst = lst


def create_tax_tree(series, zeroflag=True):
    tempGraph = nx.Graph()
    """workbook = load_workbook(filename="random_Otus.xlsx")
    sheet = workbook.active"""
    valdict = {("Bacteria",): [0,0], ("Archaea",): [0,0]}
    bac = []
    for i, (tax, val) in enumerate(series.items()):
        # adding the bacteria in every column
        bac.append(Bacteria(tax, val))
        if len(bac[i].lst) == 1 and bac[i].lst[0] == "Bacteria":
            valdict[("Bacteria",)][0] += bac[i].val
            valdict[("Bacteria",)][1] = 1
        if len(bac[i].lst) == 1 and bac[i].lst[0] == "Archaea":
            valdict[("Archaea",)][0] += bac[i].val
            valdict[("Archaea",)][1] = 1
        # connecting to the root of the tempGraph
        tempGraph.add_edge(("anaerobe",), (bac[i].lst[0],))
        # connecting all levels of the taxonomy
        for j in range(0, len(bac[i].lst) - 1):
            updateval(tempGraph, bac[i], valdict, j, 1)
        # adding the value of the last node in the chain
        updateval(tempGraph, bac[i], valdict, len(bac[i].lst) - 1, 0)
    valdict[("anaerobe",)] = [0,1]
    return create_final_graph(tempGraph, valdict, zeroflag)


def updateval(graph, bac, vald, num, adde):
    if adde == 1:
        graph.add_edge(tuple(bac.lst[:num+1]), tuple(bac.lst[:num+2]))
    # adding the value of the nodes
    if tuple(bac.lst[:num+1]) in vald:
        vald[tuple(bac.lst[:num+1])][0] += bac.val
        vald[tuple(bac.lst[:num+1])][1] = 1 - adde
    else:
        vald[tuple(bac.lst[:num+1])] = [bac.val, 1 - adde]


def create_final_graph(tempGraph, valdict, zeroflag):
    for node in nx.dfs_postorder_nodes(tempGraph, source=("anaerobe",)):
        if tempGraph.degree[node] != 1:
            valdict[node][0] = 0
        divnum = tempGraph.degree[node] + valdict[node][1] - 1
        for neigh in tempGraph.neighbors(node):
            if len(neigh) > len(node):
                valdict[node][0] += valdict[neigh][0]
        valdict[node][0] /= divnum
    valdict[("anaerobe",)][0] = (valdict[("Bacteria",)][0] + valdict[("Archaea",)][0])/2
    graph = nx.Graph()
    for e in tempGraph.edges():
        node1_name = e[0]
        node1_val = valdict[e[0]][0]
        node2_name = e[1]
        node2_val = valdict[e[1]][0]
        graph.add_node(node1_name, val=node1_val)
        graph.add_node(node2_name, val=node2_val)
        if not zeroflag or node1_val * node2_val != 0:
            graph.add_edge(node1_name, node2_name)
    return graph
