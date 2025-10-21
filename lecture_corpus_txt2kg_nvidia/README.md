1. Nvidia Docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation
2. Start Nvidia NIM on local machine: https://build.nvidia.com/nvidia/llama-3_1-nemotron-nano-4b-v1_1/deploy

3. Run python txt2kg_rag.py --NV_NIM_KEY=$YOUR_KEY --ENDPOINT_URL=http://0.0.0.0:8000/v1 --NV_NIM_MODEL nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1
4. Install pyg_lib