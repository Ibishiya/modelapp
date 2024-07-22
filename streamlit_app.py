import cv2
import torch
import numpy as np
import fitz  # PyMuPDF
import os
import uuid  # Para generar nombres únicos de carpeta
import copy
import random
import json
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoProcessor, AutoModelForCausalLM
import pandas as pd
import streamlit as st
# Cargar el modelo y el procesador
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
st.title("Modelo 1")
