from typing import NamedTuple
import logging
from langchain_ollama import OllamaEmbeddings


class Embedder:
    class HostPreEmbedding(NamedTuple):
        os: str
        port_set: str
        services: str

    class HostEmbedding(NamedTuple):
        os: list[float]
        port_set: list[float]
        services: list[float]

    def __init__(self, model_name: str):
        self.model = OllamaEmbeddings(model=model_name)

    def embed(self, normalised_host_data):
        data = self._prep_to_embed(normalised_host_data)
        if data is None:
            logging.warning(f"Could not prep to embed {normalised_host_data}")
            return

        os_embedding = self.model.embed_query(data.os)
        ports_embedding = self.model.embed_query(data.port_set)
        services_embedding = self.model.embed_query(data.services)

        return self.HostEmbedding(os_embedding, ports_embedding, services_embedding)

    def _format_service_preembedding(self, port: int, service: str) -> str:
        service_split = service.split("-")
        if service == "":
            return ""
        if len(service_split) < 2:
            return f"port {port} runs {service} service\n"

        protocol = service_split[0]
        service = service_split[1]
        return f"port {port} runs {protocol} server {service}\n"

    def _prep_to_embed(self, host):
        services = ""
        open_ports = ""
        os = ""

        if "services" in host:
            for port, service in host["services"].items():
                services += self._format_service_preembedding(port, service)

        if "open_ports" in host:
            open_ports = f"open tcp ports: {host['open_ports']}"

        if "os" in host:
            os += f"this host is os: {host['os']}\n"

        if "os_version" in host:
            os += f"version: {host['os_version']}\n"

        if "distribution" in host:
            os += f"distribution: {host['distribution']}\n"

        if "device_vendor" in host:
            os += f"device_vendor: {host['device_vendor']}"

        return self.HostPreEmbedding(
            os,
            open_ports,
            services,
        )
