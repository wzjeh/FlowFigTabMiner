# Hugging Face Space (Docker SDK, needs a PRO account): the pipeline image in
# its `web` mode.  Weights are baked in because free Spaces have no persistent
# disk.  Secrets set in the Space settings: GEMINI_API_KEY, DEMO_PASSWORD.
FROM ghcr.io/wzjeh/flowfigtabminer:latest

RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('wyzhaoc/FlowFigTabMiner-models', local_dir='/app/models'); \
    snapshot_download('yifeihu/TF-ID-base')"

# Spaces run the container as uid 1000; the pipeline writes under /app/data.
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user
ENV HOME=/home/user
ENV FFTM_THREAD_CAP=2

EXPOSE 7860
CMD ["web"]
