#!/usr/bin/env python3
"""
Linux Setup Script for Neo4j GraphRAG with GNN+LLM - Initial Setup

This script handles the initial setup steps:
1. Java installation
2. Neo4j database installation and setup
3. Python environment setup
4. Environment configuration

Usage:
    python setup_linux.py [--skip-neo4j]
"""

import argparse
import os
import sys
import subprocess
import time
import urllib.request
import tarfile

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_step(step_num, title, description=""):
    print(f"\n{Colors.HEADER}{Colors.BOLD}Step {step_num}: {title}{Colors.ENDC}")
    if description:
        print(f"{Colors.OKBLUE}{description}{Colors.ENDC}")

def print_success(message):
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")

def print_warning(message):
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")

def print_error(message):
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")

def run_command(command, check=True):
    try:
        subprocess.run(command, shell=True, check=check)
        return True
    except subprocess.CalledProcessError as e:
        if check:
            print_error(f"Command failed: {command}")
            return False
        return False

def check_command_exists(command):
    try:
        subprocess.run([command, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_java():
    print_step(1, "Installing Java", "Installing OpenJDK 17...")
    
    if check_command_exists("java"):
        print_success("Java is already installed")
        return True
    
    print("Installing OpenJDK 17...")
    run_command("sudo apt update")
    run_command("sudo apt install -y openjdk-17-jdk")
    
    # Verify installation
    if check_command_exists("java"):
        print_success("Java installed successfully")
        return True
    else:
        print_error("Java installation failed")
        return False

def install_neo4j():
    print_step(2, "Installing Neo4j", "Downloading and installing Neo4j database...")
    
    if check_command_exists("neo4j"):
        print_success("Neo4j is already installed")
        return True
    
    neo4j_version = "5.23.0"
    neo4j_file = f"neo4j-community-{neo4j_version}-unix.tar.gz"
    neo4j_url = f"https://dist.neo4j.org/neo4j-community-{neo4j_version}-unix.tar.gz"
    
    print(f"Downloading Neo4j {neo4j_version}...")
    
    # Download Neo4j if not already present
    if not os.path.exists(neo4j_file):
        print(f"Downloading from {neo4j_url}...")
        urllib.request.urlretrieve(neo4j_url, neo4j_file)
    
    # Extract Neo4j
    print("Extracting Neo4j...")
    with tarfile.open(neo4j_file, 'r:gz') as tar:
        tar.extractall()
    
    # Move to /opt
    neo4j_dir = f"neo4j-community-{neo4j_version}"
    run_command(f"sudo mv {neo4j_dir} /opt/neo4j")
    
    # Create symlink
    run_command("sudo ln -sf /opt/neo4j/bin/neo4j /usr/local/bin/neo4j")
    
    # Set permissions
    run_command("sudo chown -R $USER:$USER /opt/neo4j")
    
    # Clean up download
    if os.path.exists(neo4j_file):
        os.remove(neo4j_file)
    
    if check_command_exists("neo4j"):
        print_success("Neo4j installed successfully")
        return True
    else:
        print_error("Neo4j installation failed")
        return False

def install_neo4j_plugins():
    print_step(3, "Installing Neo4j Plugins", "Installing Graph Data Science and GenAI plugins...")
    
    # Install Graph Data Science plugin
    print("Installing Graph Data Science plugin...")
    gds_version = "2.11.0"
    gds_url = f"https://graphdatascience.ninja/neo4j-graph-data-science-{gds_version}.zip"
    gds_file = f"neo4j-graph-data-science-{gds_version}.zip"
    
    if not os.path.exists(gds_file):
        print(f"Downloading Graph Data Science plugin {gds_version}...")
        urllib.request.urlretrieve(gds_url, gds_file)
    
    # Extract and install GDS plugin
    import zipfile
    with zipfile.ZipFile(gds_file, 'r') as zip_ref:
        zip_ref.extractall("gds-temp")
    
    # Find the jar file
    gds_jar = None
    for root, dirs, files in os.walk("gds-temp"):
        for file in files:
            if file.endswith(".jar") and "graph-data-science" in file:
                gds_jar = os.path.join(root, file)
                break
        if gds_jar:
            break
    
    if gds_jar:
        run_command(f"sudo cp {gds_jar} /opt/neo4j/plugins/")
        print_success("Graph Data Science plugin installed")
    else:
        print_warning("Could not find Graph Data Science plugin jar file")
    
    # Clean up GDS temp files
    run_command("rm -rf gds-temp")
    if os.path.exists(gds_file):
        os.remove(gds_file)
    
    # Install GenAI plugin
    print("Installing GenAI plugin...")
    genai_source = "/opt/neo4j/products/neo4j-genai-plugin-5.23.0.jar"
    genai_dest = "/opt/neo4j/plugins/neo4j-genai-plugin-5.23.0.jar"
    
    if os.path.exists(genai_source):
        run_command(f"sudo cp {genai_source} {genai_dest}")
        print_success("GenAI plugin installed")
    else:
        print_warning("GenAI plugin not found in products directory")
    
    return True

def setup_neo4j():
    print_step(4, "Setting up Neo4j", "Configuring and starting Neo4j database...")
    
    # Check if Neo4j is running
    try:
        result = subprocess.run("neo4j status", shell=True, capture_output=True, text=True)
        if "running" in result.stdout.lower():
            print_success("Neo4j is already running")
            return True
    except:
        pass
    

    
    # Set initial password
    print("Setting initial password for Neo4j...")
    run_command("/opt/neo4j/bin/neo4j-admin dbms set-initial-password 12345678")
    
    # Start Neo4j
    print("Starting Neo4j...")
    run_command("/opt/neo4j/bin/neo4j start")
    
    # Wait for Neo4j to be ready
    print("Waiting for Neo4j to be ready...")
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            result = subprocess.run("neo4j status", shell=True, capture_output=True, text=True)
            if "running" in result.stdout.lower():
                print_success("Neo4j is running with plugins")
                break
        except:
            pass
        
        if attempt == max_attempts - 1:
            print_error("Neo4j failed to start within expected time")
            return False
        
        time.sleep(2)
    
    return True

def setup_python_environment():
    print_step(5, "Setting up Python Environment", "Installing Python dependencies...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor != 11:
        print_warning(f"Python 3.11 required, but found {python_version.major}.{python_version.minor}.{python_version.micro}")
        print("Installing Python 3.11...")
        
        # Install Python 3.11
        run_command("sudo apt update")
        run_command("sudo apt install -y python3.11 python3.11-pip python3.11-venv")
        
        # Set Python 3.11 as default
        run_command("sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1")
        run_command("sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1")
        
        print_success("Python 3.11 installed and set as default")
        
        # Re-execute the script with Python 3.11
        print("Restarting script with Python 3.11...")
        os.execv(sys.executable, ['python3'] + sys.argv)
    
    print_success(f"Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # Install pip if not present
    try:
        subprocess.run("python3 -m pip --version", shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("Installing pip...")
        run_command("sudo apt install -y python3-pip")
    
    # Upgrade pip
    run_command("python3 -m pip install --upgrade pip")
    
    # Install requirements
    print("Installing Python dependencies...")
    run_command("python3 -m pip install -r requirements.txt")
    
    print_success("Python dependencies installed")
    return True

def setup_environment_file():
    print_step(6, "Setting up Environment", "Creating database configuration file...")
    
    # Prompt for OpenAI API key
    print("OpenAI API Key Setup:")
    print("You can get your API key from: https://platform.openai.com/api-keys")
    print("Press Enter to skip if you don't need OpenAI models")
    
    openai_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    
    env_content = f"""NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=12345678

# OpenAI API key
"""
    
    if openai_key:
        env_content += f"OPENAI_API_KEY={openai_key}\n"
    else:
        env_content += "# OPENAI_API_KEY=your_openai_api_key_here\n"
    
    with open("db.env", "w") as f:
        f.write(env_content)
    
    print_success("Database configuration file created (db.env)")
    if openai_key:
        print_success("OpenAI API key added to db.env")
    else:
        print_warning("OpenAI API key not added. You can add it later to db.env")



def main():
    parser = argparse.ArgumentParser(description="Linux Initial Setup for Neo4j GraphRAG with GNN+LLM")
    parser.add_argument("--skip-neo4j", action="store_true", help="Skip Neo4j installation and setup")
    
    args = parser.parse_args()
    
    print(f"{Colors.HEADER}{Colors.BOLD}Neo4j GraphRAG with GNN+LLM - Initial Linux Setup{Colors.ENDC}")
    print("This script will set up the basic environment needed for the experiments.")
    
    # Check if running as root
    if os.geteuid() == 0:
        print_error("Please do not run this script as root/sudo")
        sys.exit(1)
    
    try:
        if not args.skip_neo4j:
            if not install_java():
                return
            if not install_neo4j():
                return
            if not install_neo4j_plugins():
                return
            if not setup_neo4j():
                return
        
        if not setup_python_environment():
            return
            
        setup_environment_file()
        
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}Initial setup completed successfully!{Colors.ENDC}")
        print("\nNext steps:")
        print("1. Load the dataset using stark_prime_neo4j_loading.ipynb or fetch from AWS S3 bucket gds-public-dataset/stark-prime-neo4j523 ")
        print("2. Run training script train.py")
        
    except KeyboardInterrupt:
        print_error("\nSetup interrupted by user")
    except Exception as e:
        print_error(f"\nSetup failed: {e}")

if __name__ == "__main__":
    main() 