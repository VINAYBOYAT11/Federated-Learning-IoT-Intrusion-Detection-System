# 📊 Dataset Download Guide — MQTT-IoT-IDS2020

The raw dataset files (CSV and PCAP) are **not included** in this repository because they are too large for GitHub (2–4 GB each).

## 📥 How to Download

**Dataset Source:** [MQTT-IoT-IDS2020](https://www.kaggle.com/datasets/cnrieiit/mqttset) or from the original research:

> *"MQTT-IoT-IDS2020: An MQTT-Based IoT Intrusion Detection Dataset"*

### Download Links
- 🔗 **Kaggle:** https://www.kaggle.com/datasets/cnrieiit/mqttset
- 🔗 **IEEE DataPort:** https://ieee-dataport.org/open-access/mqtt-iot-ids2020-mqtt-internet-things-intrusion-detection-dataset

---

## 📁 Required Files

Place downloaded files in the appropriate directories:

### For `experiments/` (Federated Learning)
```
experiments/
├── data/
│   ├── biflow_normal.csv
│   ├── biflow_scan_A.csv
│   ├── biflow_scan_sU.csv
│   ├── biflow_sparta.csv
│   └── biflow_mqtt_bruteforce.csv
```

### For `dataset_scripts/` (PCAP Analysis)
```
dataset_scripts/
├── data/
│   ├── normal.pcap
│   ├── normal.csv
│   ├── bruteforce.pcap
│   ├── mqtt_bruteforce.csv
│   ├── scan_A.pcap / scan_A.csv
│   ├── scan_sU.pcap / scan_sU.csv
│   └── sparta.pcap / sparta.csv
```

---

## 📊 Dataset Description

| Attack Type | Description |
|-------------|-------------|
| `normal` | Benign IoT MQTT traffic |
| `scan_A` | Active network scanning attack |
| `scan_sU` | UDP scanning attack |
| `sparta` | Brute-force dictionary attack |
| `mqtt_bruteforce` | MQTT-specific brute force attack |

### Features
- **Biflow features** — bidirectional flow statistics
- **Uniflow features** — unidirectional flow statistics
- **Packet features** — raw packet-level features
- **PCAP files** — raw network captures (Wireshark compatible)

---

## 🛠️ Pre-processing

After downloading, run the PCAP parser to extract features:

```bash
cd dataset_scripts
python pcap_parser.py
python pcap_packet_features.py
```

Then run classification experiments:
```bash
python classification.py --mode 2  # biflow features
```
