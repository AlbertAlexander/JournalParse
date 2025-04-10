def build_emotional_graph(journal_entries):
    G = nx.DiGraph()
    
    for entry in journal_entries:
        # Extract emotions and their triggers/targets
        emotions = extract_emotions(entry.text)
        
        for emotion in emotions:
            # Add emotion node if it doesn't exist
            if not G.has_node(emotion.name):
                G.add_node(emotion.name, type="emotion",
                          valence=emotion.valence,
                          arousal=emotion.arousal,
                          first_seen=entry.date,
                          entries=[entry.id])
            else:
                G.nodes[emotion.name]["entries"].append(entry.id)
            
            # Link emotion to triggers
            if emotion.trigger:
                if not G.has_node(emotion.trigger):
                    G.add_node(emotion.trigger, type="emotion_trigger",
                              entries=[entry.id])
                
                if not G.has_edge(emotion.trigger, emotion.name):
                    G.add_edge(emotion.trigger, emotion.name, 
                              type="triggers",
                              contexts=[extract_context(entry.text, emotion)],
                              entries=[entry.id])
                else:
                    edge = G.edges[emotion.trigger, emotion.name]
                    edge["contexts"].append(extract_context(entry.text, emotion))
                    edge["entries"].append(entry.id)
            
            # Link emotion to responses
            if emotion.response:
                if not G.has_node(emotion.response):
                    G.add_node(emotion.response, type="emotion_response",
                              entries=[entry.id])
                
                if not G.has_edge(emotion.name, emotion.response):
                    G.add_edge(emotion.name, emotion.response,
                              type="leads_to",
                              contexts=[extract_context(entry.text, emotion)],
                              entries=[entry.id])
                else:
                    edge = G.edges[emotion.name, emotion.response]
                    edge["contexts"].append(extract_context(entry.text, emotion))
                    edge["entries"].append(entry.id)
    
    return G
