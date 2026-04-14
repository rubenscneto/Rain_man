"""
Estimador de Baralhos — mede o montão de cartas durante a mistura.

Durante "MISTURA EM ANDAMENTO" (Evolution Gaming), o croupier
tem TODAS as cartas visíveis na mesa. Medindo a espessura do montão
e comparando com a largura da carta (que é padronizada), estimamos
quantos baralhos há no sapato.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MATEMÁTICA DO ESTIMADOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dimensões padrão de uma carta:
  Largura  = 63,5 mm
  Altura   = 88,9 mm
  Espessura = 0,30 mm

Quando N cartas são empilhadas em pé (de lado):
  Espessura do montão = N × 0,30 mm
  Largura do montão   ≈ 63,5 mm  (constante)
  Razão aspecto       = espessura / largura = N × 0,30 / 63,5

Valores esperados (razão aspecto teórica):
  4 baralhos (208 cartas): 208 × 0,30 / 63,5 = 0,983
  6 baralhos (312 cartas): 312 × 0,30 / 63,5 = 1,474
  8 baralhos (416 cartas): 416 × 0,30 / 63,5 = 1,965

Correção da câmera (Evolution Gaming filma ~30° de cima):
  Comprime a dimensão vertical em cos(30°) ≈ 0,866
  Razão observada (na tela):
    4 baralhos: 0,983 × 0,866 ≈ 0,85
    6 baralhos: 1,474 × 0,866 ≈ 1,28
    8 baralhos: 1,965 × 0,866 ≈ 1,70

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LIMITAÇÕES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Durante mistura ativa, as cartas estão espalhadas (aumenta incerteza)
- Compressão de perspectiva varia com ângulo exato da câmera
- Estimativa é ±1 baralho (suficiente para saber se é 6 ou 8)
- Média de múltiplos frames melhora precisão

Uso típico: Evolution Gaming Blackjack A → 8 baralhos S17 DAS (confirmado visualmente)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Mesmos limites HSV do evolution_detector (cartas brancas/creme)
_CARD_HSV_LOW  = np.array([0,   0,  155])
_CARD_HSV_HIGH = np.array([180, 65, 255])

# Área mínima do blob para ser considerado o montão principal
# A 1080p, 8 baralhos empilhados devem ter >10.000 px²
_MIN_STACK_AREA = 8_000
_MAX_STACK_AREA = 200_000  # evita capturar toda a mesa


@dataclass
class DeckEstimate:
    """Resultado da estimativa visual de baralhos."""
    estimated_decks: int        # Melhor estimativa: 4, 6, ou 8
    confidence: float           # 0.0–1.0
    aspect_ratio: float         # Razão altura/largura medida do montão
    stack_area_px: int          # Área em pixels do maior blob de cartas
    n_frames_averaged: int = 1  # Frames usados para média
    method: str = "visual"

    def __str__(self) -> str:
        conf_str = f"{self.confidence:.0%}"
        return (
            f"~{self.estimated_decks} baralhos estimados "
            f"(confiança: {conf_str}, razão aspecto: {self.aspect_ratio:.2f})"
        )


@dataclass
class DeckEstimator:
    """
    Acumula estimativas de múltiplos frames durante a mistura
    e retorna uma estimativa consolidada.
    """
    # Parâmetros de classificação (razão aspecto observada na câmera)
    # Pontos médios entre os valores esperados de cada configuração
    _THRESH_4_6: float = field(default=1.06, init=False, repr=False)  # entre 4 e 6 baralhos
    _THRESH_6_8: float = field(default=1.49, init=False, repr=False)  # entre 6 e 8 baralhos

    _estimates: List[DeckEstimate] = field(default_factory=list, init=False, repr=False)

    def reset(self) -> None:
        """Limpa estimativas acumuladas (usar ao início de cada mistura)."""
        self._estimates.clear()

    def add_frame(self, img: np.ndarray) -> Optional[DeckEstimate]:
        """
        Processa um frame e adiciona estimativa ao acumulador.

        Parameters
        ----------
        img : np.ndarray
            Frame RGB capturado durante "MISTURA EM ANDAMENTO".

        Returns
        -------
        DeckEstimate para este frame, ou None se nenhum montão detectado.
        """
        est = _estimate_from_single_frame(img, self._THRESH_4_6, self._THRESH_6_8)
        if est is not None:
            self._estimates.append(est)
        return est

    def consolidate(self) -> Optional[DeckEstimate]:
        """
        Retorna a estimativa final baseada em todos os frames acumulados.
        Usa voto majoritário + média das razões de aspecto.
        """
        if not self._estimates:
            return None

        # Voto majoritário
        from collections import Counter
        votes = Counter(e.estimated_decks for e in self._estimates)
        best_decks, vote_count = votes.most_common(1)[0]
        confidence_base = vote_count / len(self._estimates)

        # Média da razão aspecto e área
        avg_aspect = sum(e.aspect_ratio for e in self._estimates) / len(self._estimates)
        avg_area   = int(sum(e.stack_area_px for e in self._estimates) / len(self._estimates))

        # Confiança aumenta com mais frames concordando
        confidence = min(0.95, confidence_base * (1 + 0.05 * len(self._estimates)))

        return DeckEstimate(
            estimated_decks=best_decks,
            confidence=confidence,
            aspect_ratio=round(avg_aspect, 3),
            stack_area_px=avg_area,
            n_frames_averaged=len(self._estimates),
        )


def _estimate_from_single_frame(
    img: np.ndarray,
    thresh_4_6: float,
    thresh_6_8: float,
) -> Optional[DeckEstimate]:
    """Estimativa interna a partir de um único frame."""
    if not CV2_AVAILABLE or img is None:
        return None

    h_img, w_img = img.shape[:2]

    # 1. Segmentar regiões de cor de carta (branco/creme)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, _CARD_HSV_LOW, _CARD_HSV_HIGH)

    # 2. Operações morfológicas para consolidar o montão
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # 3. Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 4. Filtrar blobs: área mínima e posição central na tela
    candidates: List[Tuple[float, int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (_MIN_STACK_AREA <= area <= _MAX_STACK_AREA):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # O montão fica na região CENTRAL da mesa durante a mistura
        cx = x + w / 2
        cy = y + h / 2
        if cx < w_img * 0.10 or cx > w_img * 0.90:
            continue
        if cy < h_img * 0.20 or cy > h_img * 0.85:
            continue
        # Razão aspecto deve ser compatível com um montão de cartas
        aspect = h / (w + 1e-6)
        if not (0.3 < aspect < 4.0):
            continue
        candidates.append((area, x, y, w, h))

    if not candidates:
        return None

    # 5. Maior blob = montão principal
    candidates.sort(reverse=True)
    area, x, y, w, h = candidates[0]
    aspect = h / (w + 1e-6)

    # 6. Classificar pelo limiar de razão aspecto
    if aspect < thresh_4_6:
        n_decks = 4
        # Distância do limiar → confiança
        dist = abs(aspect - thresh_4_6) / thresh_4_6
        confidence = min(0.80, 0.50 + dist * 1.5)
    elif aspect < thresh_6_8:
        n_decks = 6
        dist = min(
            abs(aspect - thresh_4_6),
            abs(aspect - thresh_6_8),
        ) / (thresh_6_8 - thresh_4_6)
        confidence = min(0.75, 0.50 + dist)
    else:
        n_decks = 8
        dist = abs(aspect - thresh_6_8) / thresh_6_8
        confidence = min(0.80, 0.50 + dist * 1.5)

    # 7. Validação cruzada pela área do blob
    #    Área esperada cresce linearmente com nº de baralhos.
    #    Intervalos empíricos a 1080p (px²):
    area_ranges = {4: (8_000, 25_000), 6: (14_000, 38_000), 8: (20_000, 55_000)}
    lo, hi = area_ranges[n_decks]
    if lo <= area <= hi:
        confidence = min(0.90, confidence + 0.10)

    return DeckEstimate(
        estimated_decks=n_decks,
        confidence=confidence,
        aspect_ratio=round(aspect, 3),
        stack_area_px=int(area),
    )
