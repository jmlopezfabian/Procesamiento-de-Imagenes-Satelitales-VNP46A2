import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Configurar backend no interactivo
#configuraar backend interactivo
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple, List
from ..core.config import IMAGE_PATH, find_image_path, temp_path
from ..core.lectura import leer_radianza
from ..core.models import MedicionResultado
from ..core.utils import parse_date, extraer_geometria, left_right_coords
from ..core.downloader import find_file, download_file
from ..core.productos import obtener_producto
from .cobertura import cobertura_exacta, metricas_ponderadas, poligono_en_pixeles

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def _verificar_que_cabe(poligono, forma, municipio, cuadrante):
    """
    Comprueba que el municipio cabe entero en la imagen que se descargó.

    `SatelliteProcessor` trabaja con un cuadrante a la vez, el que se le pasa. Si
    el municipio se sale de él, el recorte se haría con índices fuera de rango
    —y en numpy un índice negativo no falla, recorta por el otro extremo—, así
    que saldría una imagen de otra parte del cuadrante con toda la pinta de ser
    la correcta. Mejor decirlo y mandar a la ruta que sí compone imágenes.
    """
    minx, miny, maxx, maxy = poligono.bounds
    alto, ancho = forma
    if minx < 0 or miny < 0 or maxx > ancho or maxy > alto:
        raise ValueError(
            f"{municipio} no cabe en {cuadrante}: su polígono llega a "
            f"x[{minx:.0f}, {maxx:.0f}] y[{miny:.0f}, {maxy:.0f}] sobre una "
            f"retícula de {ancho}x{alto}. Es un municipio repartido entre varios "
            f"cuadrantes; procésalo con ntl.radianza (process_image_mosaico), que "
            f"compone las imágenes, o con SatelliteImagesAsync."
        )


class SatelliteProcessor:
    """
    Clase principal para procesar imágenes satelitales de Black Marble (VNP46A2 o VJ146A2).
    """
    
    def __init__(self, municipio: str, producto=None):
        self.municipio = municipio
        # None = el por omisión (NTL_PRODUCTO o VJ146A2), resuelto ya para que
        # un nombre inválido falle al construir y no a media corrida.
        self.producto = obtener_producto(producto)
    
    def _save_plot(self, fig, date_obj: str, quadrant: str, plot_type: str = "analysis"):
        """
        Guarda una figura de matplotlib en lugar de mostrarla.
        
        Args:
            fig: Figura de matplotlib
            date_obj: Fecha del análisis
            quadrant: Cuadrante de la imagen
            plot_type: Tipo de gráfica
        """
        try:
            # Crear directorio temp si no existe
            filename = str(temp_path(f"{date_obj}_{self.municipio}_{quadrant}_{plot_type}.png"))
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Gráfica guardada como: {filename}")
        except Exception as e:
            print(f"Error al guardar la gráfica: {e}")
        finally:
            plt.close(fig)
    
    def _buscar_y_bajar(self, year, day, date_obj, quadrant) -> Optional[str]:
        """Ruta del .h5 descargado, o None (con el motivo impreso) si no se pudo."""
        motivo = self.producto.motivo_fuera_de_rango(date_obj)
        if motivo:
            # Sin red: la fecha no está en la vigencia del producto.
            print(f"{motivo}; no se pide a NASA.")
            return None
        h5_url = find_file(year, day, quadrant, self.producto)
        if not h5_url:
            print("No se encontró el archivo.")
            return None

        save_path = str(temp_path(
            f"{date_obj}_{self.producto.nombre}_{self.municipio}_{quadrant}.h5"))
        h5_save_path = download_file(h5_url, save_path)
        if not h5_save_path:
            print("Fallo la descarga del archivo.")
            return None
        return h5_save_path

    def get_measures(self, date_str: str, quadrant: str, show_plots: bool = True) -> Optional[dict]:
        """
        Consulta, descarga y extrae las coordenadas de una imagen satelital para un día y cuadrante.
        
        Args:
            date_str: Fecha en formato dd-mm-yy
            quadrant: Cuadrante de la imagen
            show_plots: Si mostrar las gráficas de visualización

        Returns:
            Diccionario con las mediciones o None si hay error
        """
        year, day, date_obj = parse_date(date_str)

        # Buscar y descargar archivo
        h5_save_path = self._buscar_y_bajar(year, day, date_obj, quadrant)
        if not h5_save_path:
            return None
        
        with h5py.File(h5_save_path, "r") as hdf_file:
            left_coord, right_coord = left_right_coords(hdf_file)
            if left_coord is None or right_coord is None:
                print("No se pudieron extraer las coordenadas del archivo HDF.")
                return None
            
            # Encontrar la ruta correcta a los datos de imagen
            image_path = find_image_path(hdf_file)
            if image_path not in hdf_file:
                print(f"Error: No se encontró la ruta '{image_path}' en el archivo HDF5.")
                print("Estructura del archivo HDF5:")
                def print_structure(name, obj):
                    print(name)
                hdf_file.visititems(print_structure)
                return None
            
            image_matrix, lectura = leer_radianza(hdf_file)
            coordenadas_municipio = extraer_geometria(self.municipio)
            if coordenadas_municipio is None:
                print("No se pudieron extraer las coordenadas del municipio.")
                return None

            # Crear copia para visualización
            copia_imagen = np.clip(image_matrix, 0, np.percentile(image_matrix, 99))

            if show_plots:
                fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(15, 15))
                ax[0][0].imshow(copia_imagen)
                ax[0][0].set_title(f"Imagen completa {self.municipio} - {date_obj}")
                
            try:
                # La pertenencia de cada píxel se resuelve por intersección
                # geométrica: es el mismo cálculo que alimenta la tabla
                # precalculada del procesamiento por lotes.
                poligono = poligono_en_pixeles(
                    coordenadas_municipio, left_coord, image_matrix.shape
                )
                _verificar_que_cabe(poligono, image_matrix.shape, self.municipio, quadrant)
                pesos, fila_0, columna_0 = cobertura_exacta(poligono)
                alto, ancho = pesos.shape
                imagen_recortada = image_matrix[fila_0:fila_0 + alto, columna_0:columna_0 + ancho]

                if imagen_recortada.size == 0:
                    print("La imagen recortada está vacía. Verifica las coordenadas del municipio.")
                    return None

                copia_imagen = np.clip(imagen_recortada, 0,
                                       np.nanpercentile(imagen_recortada, 99))

                # Contornos del municipio en coordenadas del recorte, para
                # superponerlos. Son varios: un municipio puede tener islas, y
                # cada hueco es un contorno más.
                contornos = []
                for parte in (poligono.geoms if hasattr(poligono, "geoms") else [poligono]):
                    for anillo in [parte.exterior, *parte.interiors]:
                        xs, ys = anillo.coords.xy
                        contornos.append((np.asarray(xs) - columna_0,
                                          np.asarray(ys) - fila_0))

                if show_plots:
                    ax[0][1].imshow(copia_imagen)
                    ax[0][1].set_title("Imagen recortada")

                    ax[1][0].imshow(copia_imagen)
                    for i, (cx, cy) in enumerate(contornos):
                        ax[1][0].plot(cx, cy, 'k-', linewidth=1.5, alpha=0.9,
                                      label='Contorno del municipio' if i == 0 else None)
                    ax[1][0].set_title("Recorte con el contorno superpuesto")
                    ax[1][0].legend(loc='upper right', fontsize=8)

                    im = ax[1][1].imshow(pesos, cmap="Blues", vmin=0, vmax=1)
                    for cx, cy in contornos:
                        ax[1][1].plot(cx, cy, 'k-', linewidth=1.5, alpha=0.9)
                    ax[1][1].set_title("Cobertura por píxel")
                    fig.colorbar(im, ax=ax[1][1], fraction=0.046)

                    ax[2][0].imshow(copia_imagen)
                    ax[2][0].imshow(pesos, cmap="Blues", alpha=0.45, vmin=0, vmax=1)
                    for cx, cy in contornos:
                        ax[2][0].plot(cx, cy, 'k-', linewidth=1.5, alpha=0.9)
                    ax[2][0].set_title("Radianza ponderada por cobertura")

                # Métricas ponderadas por área: invariantes al factor de escala
                metricas = metricas_ponderadas(imagen_recortada, pesos)

                if show_plots:
                    dentro = pesos > 0
                    if dentro.any():
                        ax[2][1].hist(imagen_recortada[dentro].astype(float), bins=50,
                                      weights=pesos[dentro], alpha=0.7,
                                      label='Píxeles del municipio', color='blue')
                        ax[2][1].grid(True)
                        ax[2][1].set_title("Histograma de radiación (ponderado por área)")
                        ax[2][1].legend()
                    else:
                        ax[2][1].text(0.5, 0.5, "No hay píxeles seleccionados",
                                    ha='center', va='center', transform=ax[2][1].transAxes)
                        ax[2][1].set_title("Sin datos")

                # Validar que hay píxeles para procesar
                if metricas is None:
                    print("No se encontraron píxeles dentro del área del municipio.")
                    return None

                # Crear medición usando solo MedicionResultado
                medicion = MedicionResultado(
                    Fecha=date_obj,
                    Producto=lectura['producto'],
                    Unidades_de_radianza=lectura['unidades'] or 'nW/(cm2 sr)',
                    **metricas,
                )
                if show_plots:
                    # Guardar la figura usando la función 
                    plt.show()
                    self._save_plot(plt.gcf(), date_obj, quadrant, "analysis")
                os.remove(h5_save_path)
                return medicion.model_dump()
                
            except Exception as e:
                print(f"Error durante el procesamiento de la imagen: {e}")
                return None

    def run(self, fechas: List[str], quadrant: str = "h08v07", show_plots: bool = False) -> pd.DataFrame:
        """
        Procesa múltiples fechas y retorna un dataframe con los resultados.
        
        Args:
            fechas: Lista de fechas en formato dd-mm-yy
            quadrant: Cuadrante de la imagen (por defecto h08v07)
            show_plots: Si mostrar las visualizaciones de matplotlib

        Returns:
            DataFrame con las mediciones de todas las fechas
        """
        results = []

        for fecha in fechas:
            print(f"Procesando fecha: {fecha}")
            try:
                datos = self.get_measures(fecha, quadrant, show_plots=show_plots)
                if datos:
                    results.append(datos)
                else:
                    print(f"No se pudieron obtener datos para {fecha}")
            except Exception as e:
                print(f"Error procesando {fecha}: {e}")
        
        if not results:
            print("No se obtuvieron datos para ninguna fecha.")
            return pd.DataFrame()
            
        return pd.DataFrame(results)

    def recorte_y_cobertura(self, date_str: str, quadrant: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Recorta la imagen y calcula la cobertura, sin agregar métricas.

        Args:
            date_str: Fecha en formato dd-mm-yy
            quadrant: Cuadrante de la imagen

        Returns:
            Tuple con (imagen_recortada, copia para visualización, cobertura por píxel)
        """
        year, day, date_obj = parse_date(date_str)

        h5_save_path = self._buscar_y_bajar(year, day, date_obj, quadrant)
        if not h5_save_path:
            return None

        with h5py.File(h5_save_path, "r") as hdf_file:
            left_coord, right_coord = left_right_coords(hdf_file)
            if left_coord is None or right_coord is None:
                print("No se pudieron extraer las coordenadas del archivo HDF.")
                return None
            
            # Encontrar la ruta correcta a los datos de imagen
            image_path = find_image_path(hdf_file)
            if image_path not in hdf_file:
                print(f"Error: No se encontró la ruta '{image_path}' en el archivo HDF5.")
                print("Estructura del archivo HDF5:")
                def print_structure(name, obj):
                    print(name)
                hdf_file.visititems(print_structure)
                return None
                
            image_matrix, _ = leer_radianza(hdf_file)
            coordenadas_municipio = extraer_geometria(self.municipio)

            if coordenadas_municipio is None:
                print("No se pudieron extraer las coordenadas del municipio.")
                return None

            copia_imagen = np.clip(image_matrix, 0, np.percentile(image_matrix, 99))
            
            try:
                poligono = poligono_en_pixeles(
                    coordenadas_municipio, left_coord, image_matrix.shape
                )
                _verificar_que_cabe(poligono, image_matrix.shape, self.municipio, quadrant)
                pesos, fila_0, columna_0 = cobertura_exacta(poligono)
                alto, ancho = pesos.shape
                imagen_recortada = image_matrix[fila_0:fila_0 + alto, columna_0:columna_0 + ancho]

                if imagen_recortada.size == 0:
                    print("La imagen recortada está vacía. Verifica las coordenadas del municipio.")
                    return None

                copia_imagen = np.clip(imagen_recortada, 0, np.nanpercentile(imagen_recortada, 99))

                return imagen_recortada, copia_imagen, pesos
                
            except Exception as e:
                print(f"Error durante el recorte de la imagen: {e}")
                return None