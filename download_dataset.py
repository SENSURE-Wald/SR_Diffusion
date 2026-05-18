from huggingface_hub import hf_hub_download

# Donwload cross-sensor dataset
hf_hub_download(repo_id="isp-uv-es/SEN2NAIP", repo_type="dataset", filename="cross-sensor/cross-sensor.zip")
