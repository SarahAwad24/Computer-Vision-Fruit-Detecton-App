import streamlit as st
from PIL import Image
import torch as torch
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av as av

webrtc_streamer(key="sample")