import pygame
import sys
import random
import pickle
import os

# --- CONFIGURACIÓN INICIAL (IDÉNTICA A TU MINIMAX.py) ---
pygame.init()

ANCHO, ALTO = 1100, 680
TAMAÑO_TABLERO = 300
TAMAÑO_CELDA = TAMAÑO_TABLERO // 3
MARGEN = 40

# Colores (Tus colores originales)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS_FONDO = (240, 240, 245)
GRIS_LINEA = (200, 200, 200)
AZUL_IA = (0, 102, 204)      # Color para la IA
ROJO_HUMANO = (204, 0, 0)    # Color para el Humano
VERDE_EXITO = (0, 153, 76)

pantalla = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Proyecto IA: Agente Q-Learning - UNEG")

# Fuentes
fuente_titulo = pygame.font.SysFont('Arial', 26, bold=True)
fuente_texto = pygame.font.SysFont('Arial', 18)
fuente_mini = pygame.font.SysFont('Arial', 14)
fuente_q = pygame.font.SysFont('Courier', 15, bold=True)

# --- CLASE AGENTE Q-LEARNING (El nuevo cerebro) ---
class AgenteQLearning:
    def __init__(self):
        self.q_table = {} 
        # Hiperparámetros agresivos para aprender en 10 partidas
        self.alpha = 0.9    # Tasa de aprendizaje alta
        self.gamma = 0.95   # Valorar mucho el futuro
        self.epsilon = 0.1  # Baja exploración (ya que entrenaremos antes)
        
    def obtener_valores_q(self, estado):
        if estado not in self.q_table:
            # Inicialización Optimista (0.5) para forzar al agente a explorar casillas
            self.q_table[estado] = [0.5] * 9
        return self.q_table[estado]

    def elegir_accion(self, tablero, disponibles):
        estado = tuple(tablero)
        qs = self.obtener_valores_q(estado)
        
        # Filtrar solo acciones disponibles
        mejores_acciones = []
        max_val = -float('inf')
        
        for d in disponibles:
            if qs[d] > max_val:
                max_val = qs[d]
                mejores_acciones = [d]
            elif qs[d] == max_val:
                mejores_acciones.append(d)
        
        return random.choice(mejores_acciones)

    def aprender(self, s, a, r, s_sig, terminal):
        qs = self.obtener_valores_q(s)
        valor_q_futuro = 0 if terminal else max(self.obtener_valores_q(s_sig))
        
        # Ecuación de Bellman (Actualización de la tabla)
        qs[a] += self.alpha * (r + self.gamma * valor_q_futuro - qs[a])

# --- CLASE JUEGO (Mantiene tu lógica de estados) ---
class TresEnRaya:
    def __init__(self):
        self.tablero = [0] * 9 # 0: Vacío, 1: Humano (X), 2: IA (O)
        self.juego_terminado = False
        self.ganador = None
        self.mensaje_estado = "Tu turno - Haz clic en una casilla"

    def reiniciar(self):
        self.__init__()

    def disponibles(self):
        return [i for i, x in enumerate(self.tablero) if x == 0]

    def verificar_estado(self):
        lineas = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for a, b, c in lineas:
            if self.tablero[a] == self.tablero[b] == self.tablero[c] != 0:
                self.ganador = self.tablero[a]
                self.juego_terminado = True
                return self.ganador
        if 0 not in self.tablero:
            self.ganador = 0 # Empate
            self.juego_terminado = True
            return 0
        return None

# --- FUNCIONES DE DIBUJO (MISMAS DE TU MINIMAX.py) ---
def dibujar_tablero_principal(juego):
    bx, by = MARGEN, 150
    pygame.draw.rect(pantalla, BLANCO, (bx, by, TAMAÑO_TABLERO, TAMAÑO_TABLERO))
    
    # Líneas
    for i in range(1, 3):
        pygame.draw.line(pantalla, GRIS_LINEA, (bx + i*TAMAÑO_CELDA, by), (bx + i*TAMAÑO_CELDA, by + TAMAÑO_TABLERO), 3)
        pygame.draw.line(pantalla, GRIS_LINEA, (bx, by + i*TAMAÑO_CELDA), (bx + TAMAÑO_TABLERO, by + i*TAMAÑO_CELDA), 3)

    # X y O
    for i, ficha in enumerate(juego.tablero):
        col = i % 3
        fil = i // 3
        cx = bx + col * TAMAÑO_CELDA + TAMAÑO_CELDA // 2
        cy = by + fil * TAMAÑO_CELDA + TAMAÑO_CELDA // 2
        
        if ficha == 1: # Humano
            pygame.draw.circle(pantalla, ROJO_HUMANO, (cx, cy), 35, 5)
        elif ficha == 2: # IA
            pygame.draw.line(pantalla, AZUL_IA, (cx-30, cy-30), (cx+30, cy+30), 5)
            pygame.draw.line(pantalla, AZUL_IA, (cx+30, cy-30), (cx-30, cy+30), 5)

def dibujar_panel_q(agente, juego):
    # Reemplazamos el Grafo por la Tabla Q actual
    rect_panel = pygame.Rect(450, 150, 600, 380)
    pygame.draw.rect(pantalla, BLANCO, rect_panel)
    pygame.draw.rect(pantalla, NEGRO, rect_panel, 2)
    
    titulo_panel = fuente_titulo.render("Memoria de Decisiones (Valores Q)", True, NEGRO)
    pantalla.blit(titulo_panel, (450, 100))
    
    estado_actual = tuple(juego.tablero)
    valores_q = agente.obtener_valores_q(estado_actual)
    
    for i, val in enumerate(valores_q):
        color_texto = AZUL_IA if val > 0.5 else (ROJO_HUMANO if val < 0.5 else NEGRO)
        txt = fuente_q.render(f"CASILLA {i}: Confianza = {val:.4f}", True, color_texto)
        pantalla.blit(txt, (470, 170 + i*35))
    
    info = fuente_mini.render(f"Estados aprendidos: {len(agente.q_table)}", True, (100,100,100))
    pantalla.blit(info, (450, 540))

# --- ENTRENAMIENTO RELÁMPAGO (10 PARTIDAS) ---
def realizar_entrenamiento(agente, num_partidas=10):
    for _ in range(num_partidas):
        temp_juego = TresEnRaya()
        while not temp_juego.juego_terminado:
            disp = temp_juego.disponibles()
            estado_prev = tuple(temp_juego.tablero)
            
            # La IA entrena contra movimientos aleatorios para ver diversos escenarios
            accion = agente.elegir_accion(temp_juego.tablero, disp)
            temp_juego.tablero[accion] = 2 # Turno IA
            
            res = temp_juego.verificar_estado()
            if res is not None:
                recompensa = 1.0 if res == 2 else (-1.5 if res == 1 else 0.2)
                agente.aprender(estado_prev, accion, recompensa, tuple(temp_juego.tablero), True)
            else:
                # Turno del "oponente imaginario"
                accion_opp = random.choice(temp_juego.disponibles())
                temp_juego.tablero[accion_opp] = 1
                res_opp = temp_juego.verificar_estado()
                recompensa = -1.5 if res_opp == 1 else 0
                agente.aprender(estado_prev, accion, recompensa, tuple(temp_juego.tablero), res_opp is not None)

# --- BUCLE PRINCIPAL ---
def main():
    agente = AgenteQLearning()
    realizar_entrenamiento(agente, 10) # <-- Entrenar antes de empezar
    juego = TresEnRaya()
    btn_rect = pygame.Rect(450, 580, 200, 50)
    
    while True:
        pantalla.fill(GRIS_FONDO)
        
        # 1. Cabecera
        texto_uni = fuente_mini.render("UNEG - Ing. Informática - Prof. Manuel Paniccia", True, (100, 100, 100))
        pantalla.blit(texto_uni, (MARGEN, 10))
        
        # 2. Dibujar Tablero y Panel Q
        dibujar_tablero_principal(juego)
        dibujar_panel_q(agente, juego)
        
        # 3. Mensaje de Estado
        msg_color = AZUL_IA if "IA" in juego.mensaje_estado else NEGRO
        msg = fuente_titulo.render(juego.mensaje_estado, True, msg_color)
        pantalla.blit(msg, (MARGEN, ALTO - 80))
        
        # 4. Botón Reiniciar
        pygame.draw.rect(pantalla, NEGRO, btn_rect)
        btn_txt = fuente_texto.render("Reiniciar Juego", True, BLANCO)
        pantalla.blit(btn_txt, (btn_rect.x + 40, btn_rect.y + 12))

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            
            if evento.type == pygame.MOUSEBUTTONDOWN:
                mx, my = evento.pos
                # Clic en Tablero (Turno Humano)
                if not juego.juego_terminado:
                    bx, by = MARGEN, 150
                    if bx < mx < bx+TAMAÑO_TABLERO and by < my < by+TAMAÑO_TABLERO:
                        col = (mx - bx) // TAMAÑO_CELDA
                        fil = (my - by) // TAMAÑO_CELDA
                        idx = fil * 3 + col
                        if juego.tablero[idx] == 0:
                            s_antes = tuple(juego.tablero)
                            juego.tablero[idx] = 1
                            res = juego.verificar_estado()
                            if res == 1:
                                agente.aprender(s_antes, idx, -1.5, tuple(juego.tablero), True)
                                juego.mensaje_estado = "¡Ganaste! (Aprendizaje incompleto)"
                            elif res == 0: juego.mensaje_estado = "¡Empate!"
                            else:
                                juego.mensaje_estado = "Turno de la IA..."
                                # TURNO DE LA IA (Inmediato)
                                pygame.display.flip()
                                pygame.time.delay(400)
                                disp_ia = juego.disponibles()
                                s_ia_antes = tuple(juego.tablero)
                                accion_ia = agente.elegir_accion(juego.tablero, disp_ia)
                                juego.tablero[accion_ia] = 2
                                res_ia = juego.verificar_estado()
                                if res_ia == 2:
                                    agente.aprender(s_ia_antes, accion_ia, 1.0, tuple(juego.tablero), True)
                                    juego.mensaje_estado = "¡La IA ha ganado!"
                                elif res_ia == 0: juego.mensaje_estado = "¡Empate!"
                                else:
                                    agente.aprender(s_ia_antes, accion_ia, 0, tuple(juego.tablero), False)
                                    juego.mensaje_estado = "Tu turno..."

                if btn_rect.collidepoint(mx, my):
                    juego.reiniciar()

        pygame.display.flip()

if __name__ == "__main__":
    main()
