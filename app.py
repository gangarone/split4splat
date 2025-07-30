import streamlit as st
import imageio                  # per leggere video via FFmpeg
import py360convert
import numpy as np
from PIL import Image, ImageDraw
import math
import io
import zipfile
import os
import base64

# Configurazione della pagina con cache migliorata
st.set_page_config(page_title="Split 4 Splat", layout="wide")

# -----------------------------------------------------------------------------
# Helper functions con cache
# -----------------------------------------------------------------------------
@st.cache_data
def to_rgba(arr: np.ndarray) -> np.ndarray:
    if arr.shape[2] == 4:
        return arr
    h, w, _ = arr.shape
    alpha = np.full((h, w, 1), 255, dtype=arr.dtype)
    return np.concatenate([arr, alpha], axis=2)

@st.cache_data
def create_checkboard(width: int, height: int, tile_size: int = 16) -> Image.Image:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    c1, c2 = (220, 220, 220), (192, 192, 192)
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            fill = c1 if ((x//tile_size)+(y//tile_size)) % 2 == 0 else c2
            draw.rectangle([x, y, x+tile_size, y+tile_size], fill=fill)
    return img

@st.cache_data
def make_preview(e_img: np.ndarray, scale_div: int = 12) -> Image.Image:
    preview = Image.fromarray(to_rgba(e_img))
    h, w = preview.height, preview.width
    preview = preview.resize((max(1, w//scale_div), max(1, h//scale_div)), Image.Resampling.BILINEAR)
    bg = create_checkboard(preview.width, preview.height).convert("RGBA")
    bg.paste(preview, (0, 0), preview)
    return bg

@st.cache_data
def get_perspective(e_img: np.ndarray, fov: float, yaw: float, pitch: float, res: int) -> np.ndarray:
    return py360convert.e2p(e_img, fov_deg=fov, u_deg=yaw, v_deg=pitch, out_hw=(res, res))

@st.cache_data
def load_image_cached(file_path: str):
    """Cache per il caricamento delle immagini"""
    return np.array(Image.open(file_path).convert("RGBA"))

@st.cache_data
def compute_auto_resolution(w: int, h: int, fov: float) -> int:
    max_w = w * (fov / 360)
    vfov = 2*math.atan(math.tan(math.radians(fov)/2)*(h/w))
    max_h = h * (math.degrees(vfov) / 180)
    return max(1, int(min(max_w, max_h)))

# Funzioni SVG ottimizzate con cache
@st.cache_data
def draw_interactive_top_view_svg(splits: int, fov: float, selected_idx: int, enabled_cams_tuple: tuple):
    """Crea solo la parte SVG della top view con numeri intorno al cerchio"""
    enabled_cams = list(enabled_cams_tuple)  # Converte da tuple a list per la logica
    size = 240
    center = size // 2
    radius = 80
    half_fov = math.radians(fov / 2)
    
    svg_elements = []
    svg_elements.append(f'<circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="lightgray" stroke-width="2"/>')
    
    for i in range(splits):
        ang = i * 2 * math.pi / splits - math.pi/2
        x1 = center + radius * math.cos(ang - half_fov)
        y1 = center + radius * math.sin(ang - half_fov)
        x2 = center + radius * math.cos(ang + half_fov)
        y2 = center + radius * math.sin(ang + half_fov)
        
        if i == selected_idx and i in enabled_cams:
            color = '#FBB000AA'
        elif i in enabled_cams:
            color = '#899DDD80'
        else:
            color = '#CCCCCC60'
        
        large_arc = 1 if (2 * half_fov) > math.pi else 0
        path_d = f"M {center} {center} L {x1} {y1} A {radius} {radius} 0 {large_arc} 1 {x2} {y2} Z"
        svg_elements.append(f'<path d="{path_d}" fill="{color}"/>')
        
        label_radius = radius + 25
        label_x = center + label_radius * math.cos(ang)
        label_y = center + label_radius * math.sin(ang)
        camera_number = i + 1
        
        if i == selected_idx and i in enabled_cams:
            text_color = '#FF8800'
        elif i in enabled_cams:
            text_color = '#333333'
        else:
            text_color = '#999999'
        
        svg_elements.append(f'''
            <circle cx="{label_x}" cy="{label_y}" r="12" fill="rgba(255,255,255,0.9)" stroke="{text_color}" stroke-width="1"/>
            <text x="{label_x}" y="{label_y + 4}" text-anchor="middle" font-family="Arial, sans-serif" 
                  font-size="11" font-weight="bold" fill="{text_color}">
                {camera_number}
            </text>
        ''')
    
    svg_content = f'''
    <svg width="{size}" height="{size}" style="background: transparent;">
        {"".join(svg_elements)}    </svg>
    '''
    
    return svg_content, size, center, radius

@st.cache_data
def draw_side_view_svg(fov: float, pitch: float):
    """Crea la side view usando SVG"""
    size = 240
    center = size // 2
    radius = 80
    
    group_elements = []
    group_elements.append(f'<circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="lightgray" stroke-width="2"/>')
    
    vfov = 2 * math.atan(math.tan(math.radians(fov)/2))
    pitch_rad = math.radians(pitch)
    
    x1 = center + radius * math.cos(pitch_rad - vfov/2)
    y1 = center + radius * math.sin(pitch_rad - vfov/2)
    x2 = center + radius * math.cos(pitch_rad + vfov/2)
    y2 = center + radius * math.sin(pitch_rad + vfov/2)
    
    large_arc = 1 if vfov > math.pi else 0
    path_d = f"M {center} {center} L {x1} {y1} A {radius} {radius} 0 {large_arc} 1 {x2} {y2} Z"
    group_elements.append(f'<path d="{path_d}" fill="#FBB000AA"/>')
    group_elements.append(f'<line x1="{center - radius}" y1="{center}" x2="{center + radius}" y2="{center}" stroke="#888888" stroke-width="1" stroke-dasharray="3,3"/>')
    
    svg_content = f'''
    <svg width="{size}" height="{size}" style="background: transparent;">
        <g transform="rotate(180, {center}, {center})">
            {"".join(group_elements)}
        </g>    </svg>
    '''
    
    return svg_content

# Funzione per caricare immagini dalla cartella root con cache
@st.cache_data
def load_root_images():
    """Carica automaticamente le immagini dalla cartella root"""
    root_folder = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    root_images = []
    
    try:
        for file in os.listdir(root_folder):
            if any(file.endswith(ext) for ext in image_extensions):
                root_images.append(os.path.join(root_folder, file))
    except:
        pass
    
    return sorted(root_images)

# Inizializzazione session state centralizzata
def init_session_state():
    """Inizializza tutti gli stati della sessione"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.camera = 0
        st.session_state.enabled_cams = list(range(5))  # Default 5 splits
        st.session_state.prev_splits = 5
        st.session_state.last_frame_idx = 0
        st.session_state.last_yaw_offset = 0

init_session_state()

# CSS per ridurre il flashing
st.markdown("""
<style>
    .stSlider > div > div > div > div {
        transition: all 0.1s ease-in-out;
    }
    .stButton > button {
        transition: all 0.1s ease-in-out;
    }
    .stMultiSelect > div {
        transition: all 0.1s ease-in-out;
    }
    /* Riduce l'animazione di ricaricamento */
    .stApp > div {
        transition: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Streamlit App

# Sposta il titolo nella sidebar
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 0.5em;'>
        <span style='font-size:2.2em; font-weight: bold;'>Split 4 Splat</span><br>
        <span style='font-size:1em; color: #888;'>equirectangular 2 persp splitter for gaussian splatting</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Input handling
items = st.sidebar.file_uploader("Drop images here (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=True, key="uploader")

if not items:
    root_images = load_root_images()
    if root_images:
        st.sidebar.info(f"Using {len(root_images)} demo images")
        items = root_images
        is_video = False
        total = len(items)
    else:
        st.sidebar.info("Please upload at least one image or place images in the root folder.")
        st.stop()
else:
    is_video = False
    total = len(items)

# Sidebar separatore
st.sidebar.markdown("---")

# Sidebar: Process frames button
process_button = st.sidebar.button("Process frames", key="process_frames_sidebar")

# Sidebar: Placeholder per progress bar
progress_container = st.sidebar.empty()

# Sidebar: Placeholder per success message
success_container = st.sidebar.empty()

# Sidebar: Placeholder per download button
download_container = st.sidebar.empty()

# Layout principale a griglia 2x2
col1, col2 = st.columns([1, 1])

with col1:
   # QUADRANTE IN ALTO A SINISTRA: Equirectangular Preview (più compatta)
   st.markdown("<h3 style='text-align: center;'>Equirectangular Preview</h3>", unsafe_allow_html=True)
   with st.container():
       preview_placeholder = st.empty()
  
   # QUADRANTE IN BASSO A SINISTRA: Controlli
   st.subheader("Controls")
  
   # Frame index
   if total >= 1:
       if total == 1:
           frame_idx = 0
           st.text("Frame index: 0 (single frame)")
       else:
           frame_idx = st.slider("Frame index", 0, max(0, total - 1), 0, key="frame_slider")
   else:
       st.warning("No frames available yet.")
       st.stop()
  
   # Yaw offset
   yaw_offset = st.slider("Yaw offset (°)", 0, 359, 0, key="yaw_slider")
  
   # Number of splits
   splits = st.slider("Number of splits/cameras", 1, 32, 5, key="splits_slider")
   
   # Gestione splits change ottimizzata
   if st.session_state.prev_splits != splits:
       st.session_state.prev_splits = splits
       st.session_state.camera = 0
       st.session_state.enabled_cams = list(range(splits))
  
   # FOV
   fov = st.slider("Fielf Of View°", 20, 160, 90, key="fov_slider")
  
   # Gestione click sulle camere dalla top view usando bottoni semplici
   def create_camera_buttons_optimized(splits, selected_idx, enabled_cams):
       """Crea una griglia di bottoni circolari per selezionare le camere"""
       st.markdown("""
       <style>
       div[data-testid="column"] > div > div > button {
           width: 32px !important;
           height: 32px !important;
           border-radius: 50% !important;
           padding: 0 !important;
           margin: 1px !important;
           font-size: 11px !important;
           font-weight: bold !important;
           display: flex !important;
           align-items: center !important;
           justify-content: center !important;
       }
       </style>
       """, unsafe_allow_html=True)
       
       # Crea griglia di bottoni - più compatta
       cols_per_row = min(6, splits)  # Ridotto a 6 bottoni per riga per risparmiare spazio
       num_rows = (splits + cols_per_row - 1) // cols_per_row
       
       for row in range(num_rows):
           # Centra ogni riga di bottoni
           if splits <= cols_per_row:
               # Se tutti i bottoni stanno in una riga, centra
               padding_cols = (cols_per_row - splits) // 2
               cols = st.columns([0.5] * padding_cols + [1] * splits + [0.5] * (cols_per_row - splits - padding_cols))
               start_idx = padding_cols
           else:
               cols = st.columns(cols_per_row)
               start_idx = 0
           
           for col_idx in range(cols_per_row):
               cam_idx = row * cols_per_row + col_idx
               if cam_idx < splits:
                   with cols[start_idx + col_idx if splits <= cols_per_row else col_idx]:
                       camera_number = cam_idx + 1
                       
                       # Determina il tipo di bottone in base allo stato
                       if cam_idx == selected_idx and cam_idx in enabled_cams:
                           button_type = "primary"  # Arancione per selezionata e abilitata
                       elif cam_idx in enabled_cams:
                           button_type = "secondary"  # Grigio per abilitata ma non selezionata  
                       else:
                           button_type = "secondary"  # Grigio per disabilitata
                       
                       # Bottone cliccabile
                       if st.button(
                           str(camera_number), 
                           key=f"topview_cam_{cam_idx}",
                           type=button_type,
                           help=f"Select Camera {camera_number}",
                           use_container_width=True
                       ):
                           st.session_state.camera = cam_idx
                           # Se la camera selezionata non è abilitata, abilitala
                           if cam_idx not in st.session_state.enabled_cams:
                               st.session_state.enabled_cams.append(cam_idx)
                               st.session_state.enabled_cams.sort()
                           return True  # Cambia da st.rerun() a return True
                   
       return False  # Aggiungi questo return
  
   # Enable cameras (1-based labels) - Usa session_state
   camera_labels = [f"Camera {i+1}" for i in range(splits)]
   enabled_labels_default = [f"Camera {i+1}" for i in st.session_state.enabled_cams if i < splits]
   
   enabled_cams_display = st.multiselect(
       "Enable cameras", 
       options=camera_labels, 
       default=enabled_labels_default,
       key="enabled_cams_multiselect"
   )
   
   # Aggiorna session_state.enabled_cams basandosi sulla selezione
   st.session_state.enabled_cams = [int(label.split()[1]) - 1 for label in enabled_cams_display]

with col2:
   st.markdown("<h3 style='text-align: center;'>Camera Preview</h3>", unsafe_allow_html=True)
   
   # Camera preview centrata
   camera_preview_placeholder = st.empty()
   
   # Spaziatura
   st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
   
   # Bottom right: Top View and Side View in layout migliorato
   view_col1, view_col2 = st.columns([1, 1])
   
   # TOP VIEW COLUMN
   with view_col1:
       st.markdown(
           "<div style='text-align: center; margin-bottom: 10px;'><b>Top View</b></div>",
           unsafe_allow_html=True,
       )
       top_view_placeholder = st.empty()
   
   # SIDE VIEW COLUMN  
   with view_col2:
       st.markdown(
           "<div style='text-align: center; margin-bottom: 10px;'><b>Side View</b></div>",
           unsafe_allow_html=True,
       )
       side_view_placeholder = st.empty()
       
       # Controllo Pitch centrato sotto la side view
       st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
       pitch_col1, pitch_col2, pitch_col3 = st.columns([0.5, 2, 0.5])
       with pitch_col2:
           pitch = st.slider("Pitch°", -90, 90, 0, key="pitch_control")

# Elaborazione dell'immagine e aggiornamento dei placeholder
if items:
   # Controlla se ci sono stati cambiamenti significativi
   params_changed = (
       st.session_state.last_frame_idx != frame_idx or
       st.session_state.last_yaw_offset != yaw_offset
   )
   
   if params_changed or 'current_image' not in st.session_state:
       # Carica l'immagine solo se necessario
       if isinstance(items[frame_idx], str):
           arr_raw = load_image_cached(items[frame_idx])
       else:
           arr_raw = np.array(Image.open(items[frame_idx]).convert("RGBA"))
       
       shift_px = int(yaw_offset * arr_raw.shape[1] / 360)
       arr_shifted = np.roll(arr_raw, shift_px, axis=1)
       e_img = to_rgba(arr_shifted)
       
       # Salva nell'session state per evitare ricalcoli
       st.session_state.current_image = arr_raw
       st.session_state.current_e_img = e_img
       st.session_state.last_frame_idx = frame_idx
       st.session_state.last_yaw_offset = yaw_offset
   else:
       # Usa i dati cached
       arr_raw = st.session_state.current_image
       e_img = st.session_state.current_e_img
   
   # Aggiorna preview solo se necessario
   if params_changed or 'preview_image' not in st.session_state:
       img = make_preview(e_img)
       buf = io.BytesIO()
       img.save(buf, format="PNG")
       img_b64 = base64.b64encode(buf.getvalue()).decode()
       st.session_state.preview_image = img_b64
   
   with preview_placeholder.container():
       st.markdown(
           f"<div style='display: flex; justify-content: center;'>"
           f"<img src='data:image/png;base64,{st.session_state.preview_image}' width='500'/>"
           f"</div>",
           unsafe_allow_html=True,
       )
   
   # Camera preview ottimizzata
   yaw_cam = (st.session_state.camera * 360 / splits + yaw_offset) % 360
   auto_res = compute_auto_resolution(e_img.shape[1], e_img.shape[0], fov)
   preview_res = max(1, int(auto_res * 0.9))
   
   persp = get_perspective(arr_raw, fov, yaw_cam, pitch, preview_res)
   img = Image.fromarray(to_rgba(persp))
   bg = create_checkboard(preview_res, preview_res).convert("RGBA")
   bg.paste(img, (0, 0), img)
   
   with camera_preview_placeholder:
       preview_col1, preview_col2, preview_col3 = st.columns([1, 3, 1])
       with preview_col2:
           st.image(bg, caption=f"Yaw {yaw_cam:.1f}°, Pitch {pitch}°")
   
   # Aggiorna le viste con cache
   enabled_cams_tuple = tuple(st.session_state.enabled_cams)  # Converte per cache
   svg_content, size, center, radius = draw_interactive_top_view_svg(splits, fov, st.session_state.camera, enabled_cams_tuple)
   
   with top_view_placeholder.container():
       st.markdown(
           f"<div style='display: flex; justify-content: center;'>{svg_content}</div>",
           unsafe_allow_html=True
       )
       
       # Spaziatura ridotta
       st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
       st.markdown("<div style='text-align: center; font-size: 11px; color: #666; margin-bottom: 6px;'>Click to select camera:</div>", unsafe_allow_html=True)
       
       camera_changed = create_camera_buttons_optimized(splits, st.session_state.camera, st.session_state.enabled_cams)
       if camera_changed:
           st.rerun()
   
   with side_view_placeholder.container():
       rotated_svg_content = draw_side_view_svg(fov, pitch)
       st.markdown(
           f"<div style='display: flex; justify-content: center;'>{rotated_svg_content}</div>",
           unsafe_allow_html=True
       )

# Calcolo e visualizzazione del numero di tiles nella sidebar
total_tiles = total * len(st.session_state.enabled_cams)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Tiles to generate:** {total} frames × {len(st.session_state.enabled_cams)} cameras = **{total_tiles} tiles**")

# Logica di elaborazione
if process_button:
   mem = io.BytesIO()
   with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
       count = 0
       total_ops = total * len(st.session_state.enabled_cams)
       for i in range(total):
           base = (os.path.splitext(vid)[0] + f"_frame{i:04d}") if is_video else (
               os.path.splitext(os.path.basename(items[i]))[0] if isinstance(items[i], str)
               else os.path.splitext(items[i].name)[0]
           )
           if is_video:
               arr_i = reader.get_data(items[i])
           else:
               if isinstance(items[i], str):
                   arr_i = load_image_cached(items[i])
               else:
                   arr_i = np.array(Image.open(items[i]).convert("RGBA"))
            
           shift_val = int(yaw_offset * arr_i.shape[1] / 360)
           rgba = to_rgba(np.roll(arr_i, shift_val, axis=1))
           for c in st.session_state.enabled_cams:
               yaw_c = (c * 360 / splits + yaw_offset) % 360
               tile = get_perspective(rgba, fov, yaw_c, pitch, compute_auto_resolution(rgba.shape[1], rgba.shape[0], fov))
               buf = io.BytesIO()
               Image.fromarray(to_rgba(tile)).save(buf, format="PNG")
               zf.writestr(f"{base}_cam{c+1:02d}.png", buf.getvalue())
               count += 1
               progress_container.progress(count / total_ops)
   
   mem.seek(0)
   # Show the success message and download button in order
   success_container.success("Processing completed!")
   download_container.download_button("Download ZIP of tiles", mem.getvalue(), file_name="tiles.zip", mime="application/zip")
