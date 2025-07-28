import yaml

def add_clients(input_file, output_file, num_clients=100):
    with open(input_file, 'r') as file:
        data = yaml.safe_load(file)
    
    for i in range(1, num_clients + 1):
        site = {
            'name': f'site-{i}',
            'type': 'client',
            'org': 'nvidia'
        }
        data['participants'].append(site)
    
    with open(output_file, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False, indent=2)
    
    print(f"{num_clients} clients added successfully!")

input_file = 'project_original.yml'
output_file = 'project.yml'

add_clients(input_file, output_file)
