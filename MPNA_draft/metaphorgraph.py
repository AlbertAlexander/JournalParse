def build_metaphor_graph(journal_entries):
    G = nx.Graph()
    
    for entry in journal_entries:
        # Extract metaphors and their targets
        metaphors = extract_metaphors(entry.text)
        for metaphor in metaphors:
            source = metaphor.source  # The concrete domain
            target = metaphor.target  # The abstract concept
            
            # Add nodes if they don't exist
            if not G.has_node(source):
                G.add_node(source, type="metaphor_source", 
                          domain=classify_domain(source),
                          entries=[entry.id])
            else:
                G.nodes[source]["entries"].append(entry.id)
                
            if not G.has_node(target):
                G.add_node(target, type="metaphor_target",
                          entries=[entry.id])
            else:
                G.nodes[target]["entries"].append(entry.id)
            
            # Add edge between source and target
            if not G.has_edge(source, target):
                G.add_edge(source, target, type="metaphorical_mapping",
                          first_seen=entry.date,
                          contexts=[extract_context(entry.text, metaphor)],
                          entries=[entry.id])
            else:
                edge = G.edges[source, target]
                edge["contexts"].append(extract_context(entry.text, metaphor))
                edge["entries"].append(entry.id)
    
    return G