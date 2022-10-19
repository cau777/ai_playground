# TODO: production mode
FROM rust:latest as builder
WORKDIR /app

RUN mkdir codebase
RUN mkdir server

WORKDIR /app/codebase
COPY codebase/Cargo.toml .
# Download and compile codebase dependencies
RUN mkdir src && echo "// dummy file" > src/lib.rs && cargo build && rm src/lib.rs

WORKDIR /app/server
COPY server/Cargo.toml .
# Removes codebase from dependencies
RUN sed -i 's|codebase = .*||g' Cargo.toml
# Download and compile server dependencies
RUN mkdir src && echo "fn main(){}" > src/main.rs && cargo build && rm src/main.rs

# Copy the rest of the files
WORKDIR /app/codebase
COPY codebase/build.rs .
COPY codebase/src ./src

WORKDIR /app/server
COPY server/Cargo.toml .
COPY server/src ./src
RUN cargo build

FROM debian:bullseye-slim as runner
WORKDIR /app
RUN mkdir build
COPY --from=builder /app/server/target/debug /app/build
WORKDIR /app/build

ENV ROCKET_ADDRESS=0.0.0.0
ENV MODEL_FILE_PATH="/app/files"
ENV AZURE_FS_URL="https://aiplaygroundmodels.file.core.windows.net"
ENV AZURE_FS_QUERY_SAS="sv=2021-06-08&ss=f&srt=o&sp=rl&se=2030-10-11T07:34:07Z&st=2022-10-10T23:34:07Z&sip=0.0.0.0-255.255.255.255&spr=https&sig=WqIPvmNfe52nD3KomqyRh9c40ftJHdCSIEMLCtTRxIM%3D"
ENV IS_LOCAL_SERVER="TRUE"

EXPOSE 8000
RUN mkdir /app/files
RUN chmod -R 777 /app/files
COPY config/digits /app/files/digits
CMD ["./server"]
#ENTRYPOINT ["tail", "-f", "/dev/null"]