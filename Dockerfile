# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.9

# Set proper Python threading configuration
# ENV OMP_NUM_THREADS=4
# ENV NUMEXPR_NUM_THREADS=4
# ENV MKL_NUM_THREADS=4

# Set timezone
ENV TZ=Asia/Jakarta
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update
RUN apt --yes install ffmpeg

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app
# CMD ["gunicorn", "--timeout", "300", "-b", "0.0.0.0:7860", "app:app"]
# CMD ["gunicorn", "--timeout", "300", "--workers", "1", "--threads", "4", "-b", "0.0.0.0:7860", "app:app"]
CMD ["gunicorn", "--timeout", "300", "--workers", "2", "--threads", "4", "--worker-class", "gthread", "-b", "0.0.0.0:7860", "app:app"]