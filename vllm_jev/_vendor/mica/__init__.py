"""Mica runtime from akivet/Mica, SDK commit f6bccd76.

Apache-2.0; see LICENSE and NOTICE. Only direct_backend.py is modified:
Windows DLL-directory handles remain alive until close, failed construction
releases native resources, and the applied runtime settings are observable.
Prompt rendering, codebook, calibration, and prefill/readout math are upstream.
"""
