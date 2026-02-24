curl https://sdk.cloud.google.com | bash
mkdir val2017-images
gsutil -m rsync gs://images.cocodataset.org/val2017 val2017-images
mkdir val2017-annotations
gsutil -m rsync gs://images.cocodataset.org/annotations val2017-annotations