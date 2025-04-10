def integrate_graphs(graphs):
    # Create a merged graph
    integrated = nx.MultiDiGraph()
    
    # Add all nodes and edges from each graph
    for i, graph in enumerate(graphs):
        graph_name = graph_names[i]
        
        for node, data in graph.nodes(data=True):
            if not integrated.has_node(node):
                integrated.add_node(node, **data)
                integrated.nodes[node]["sources"] = [graph_name]
            else:
                # Merge node attributes
                for key, value in data.items():
                    if key in integrated.nodes[node]:
                        if isinstance(value, list):
                            integrated.nodes[node][key].extend(value)
                        else:
                            # For non-list values, keep most recent
                            integrated.nodes[node][key] = value
                    else:
                        integrated.nodes[node][key] = value
                
                if graph_name not in integrated.nodes[node]["sources"]:
                    integrated.nodes[node]["sources"].append(graph_name)
        
        # Add edges with their graph source
        for u, v, data in graph.edges(data=True):
            integrated.add_edge(u, v, **data, source=graph_name)
    
    # Create cross-references between graphs
    # Link identical entities across graphs
    nodes = list(integrated.nodes())
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            # If nodes appear in different graphs and are similar
            if (set(integrated.nodes[node1]["sources"]) != 
                set(integrated.nodes[node2]["sources"]) and
                calculate_node_similarity(node1, node2, integrated) > 0.8):
                
                integrated.add_edge(node1, node2, type="cross_reference",
                                   similarity=calculate_node_similarity(node1, node2, integrated))
    
    return integrated
