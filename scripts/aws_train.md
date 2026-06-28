# Training the predictor on AWS (within the credit budget)

This recipe launches a multi-GPU instance, runs the ESM2 fine-tune, pulls the
checkpoint back, and tears the instance down. Quota check (already verified for
this account): `aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA`
returned 384 vCPUs for G/VT, so `g5.12xlarge` (4x A10G, 48 vCPU) is available.

## 1. Launch a GPU instance

Use the AWS Deep Learning AMI (PyTorch) so CUDA + drivers are preinstalled. Find
the latest AMI id, then launch:

```bash
REGION=us-west-2
# Latest Deep Learning OSS PyTorch AMI (Ubuntu 22.04). Look it up:
AMI=$(aws ssm get-parameters --region $REGION \
  --names /aws/service/deeplearning/ami/x86_64/oss-pytorch/latest/ubuntu-22.04 \
  --query 'Parameters[0].Value' --output text 2>/dev/null || echo "ami-xxxxxxxx")

aws ec2 run-instances --region $REGION \
  --image-id "$AMI" \
  --instance-type g5.12xlarge \
  --key-name YOUR_KEY \
  --security-group-ids sg-xxxxxxxx \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=coscientist-train}]' \
  --instance-market-options '{"MarketType":"spot"}'   # optional: spot to save credits
```

## 2. Train

```bash
ssh ubuntu@<PUBLIC_IP>
git clone <YOUR_REPO> && cd ai-coscientist
pip install -r requirements-protein.txt
# DDP, ESM2-35M, 4 GPUs:
./training/launch_multi_gpu.sh 4 ddp facebook/esm2_t12_35M_UR50D
# or FSDP, ESM2-650M:
./training/launch_multi_gpu.sh 4 fsdp facebook/esm2_t33_650M_UR50D
python -m training.plot_curves
```

## 3. Pull the checkpoint back

```bash
scp -r ubuntu@<PUBLIC_IP>:~/ai-coscientist/checkpoints/esm2_fitness ./checkpoints/
```

## 4. Tear down (stop the spend)

```bash
aws ec2 terminate-instances --region us-west-2 --instance-ids i-xxxxxxxx
```

## Cost note

`g5.12xlarge` on-demand is ~$5.7/hr in us-west-2; the synthetic/35M run finishes
in well under an hour. Even a few hours of experimentation is a tiny fraction of
the $1000 credit budget. Use `--instance-market-options` spot to cut it further.
