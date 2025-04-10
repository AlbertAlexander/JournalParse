def build_relationship_graph(journal_entries):
    G = nx.DiGraph()
    
    # Extract people mentioned
    for entry in journal_entries:
        people = extract_people(entry.text)
        for person in people:
            # Create or update person node
            if not G.has_node(person):
                G.add_node(person, type="person", first_mention=entry.date, 
                          mentions=1, entries=[entry.id])
            else:
                G.nodes[person]["mentions"] += 1
                G.nodes[person]["entries"].append(entry.id)
                
            # Extract relationship between writer and person
            sentiments = extract_relationship_sentiment(entry.text, person)
            interactions = extract_interactions(entry.text, person)
            
            # Add relationship edge with writer
            if not G.has_edge("SELF", person):
                G.add_edge("SELF", person, type="relationship", 
                          first_mention=entry.date,
                          sentiments=[sentiments],
                          interactions=[interactions],
                          entries=[entry.id])
            else:
                edge = G.edges["SELF", person]
                edge["sentiments"].append(sentiments)
                edge["interactions"].append(interactions)
                edge["entries"].append(entry.id)
            
            # Add relationships between mentioned people
            other_people = [p for p in people if p != person]
            for other in other_people:
                rel = extract_described_relationship(entry.text, person, other)
                if rel and not G.has_edge(person, other):
                    G.add_edge(person, other, type="described_relationship",
                              description=rel, entries=[entry.id])
                elif rel:
                    G.edges[person, other]["entries"].append(entry.id)
    
    return G