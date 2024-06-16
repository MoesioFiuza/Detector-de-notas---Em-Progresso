import sys
import os
import numpy as np
import sounddevice as sd
import librosa
import subprocess
from music21 import stream, note, environment
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QVBoxLayout, QWidget, QLabel, QGraphicsView, QGraphicsScene, QFileDialog, QGraphicsPixmapItem)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QPointF, QPropertyAnimation, pyqtProperty
from PyQt5.QtGui import QPixmap, QFont, QPainter, QColor, QBrush
from qt_material import apply_stylesheet
import random

# Configura o caminho para o executável do LilyPond
environment.UserSettings()['lilypondPath'] = r'D:\lily\lilypond-2.24.3\bin\lilypond.exe'

# Função para converter frequência em nota musical
def freq_para_nota(freq):
    A4 = 440
    C0 = A4 * pow(2, -4.75)
    nome = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    h = round(12 * np.log2(freq / C0))
    oitava = h // 12
    n = h % 12
    return nome[n] + str(oitava)

class ProcessadorAudio(QThread):
    nota_detectada = pyqtSignal(note.Note)
    captura_finalizada = pyqtSignal()

    def __init__(self, taxa=44100):
        super().__init__()
        self.taxa = taxa
        self.partitura_musical = stream.Stream()
        self.buffer_dados = []

    def run(self):
        def callback(dados_entrada, frames, tempo, status):
            if status:
                print(status)
            self.buffer_dados.extend(dados_entrada[:, 0])

        with sd.InputStream(callback=callback, channels=1, samplerate=self.taxa, device='loopback'):
            sd.sleep(0)  # Mantém o stream aberto até ser explicitamente parado

    def parar(self):
        self.terminate()
        self.processar_notas()
        self.captura_finalizada.emit()

    def processar_notas(self):
        dados = np.array(self.buffer_dados)
        frequencias, magnitudes = librosa.core.piptrack(y=dados, sr=self.taxa)
        for t in range(frequencias.shape[1]):
            indice = magnitudes[:, t].argmax()
            frequencia = frequencias[indice, t]
            if frequencia > 0:
                nome_nota = freq_para_nota(frequencia)
                nota_musical = note.Note(nome_nota)
                self.partitura_musical.append(nota_musical)
                self.nota_detectada.emit(nota_musical)

class VisaoCaleidoscopio(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.cena = QGraphicsScene(self)
        self.setScene(self.cena)
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30), Qt.SolidPattern))
        self.setMinimumSize(400, 400)
        self.temporizador = QTimer()
        self.temporizador.timeout.connect(self.atualizar)
        self.temporizador.start(50)
        self.formas = self.gerar_formas()
        self._t = 0
        self.transformacao = 'caleidoscopio'  # Alternar entre 'caleidoscopio' e 'headset'
        self.animacao = QPropertyAnimation(self, b't')

    def gerar_formas(self):
        formas = []
        for i in range(50):
            cor = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            posicao_inicial = QPointF(random.uniform(0, self.width()), random.uniform(0, self.height()))
            tamanho = random.uniform(20, 50)
            formas.append({
                'cor': cor,
                'posicao_inicial': posicao_inicial,
                'tamanho': tamanho,
            })
        return formas

    def iniciar_animacao(self):
        self.animacao.setDuration(10000)
        self.animacao.setStartValue(0)
        self.animacao.setEndValue(2 * np.pi)
        self.animacao.start()

    @pyqtProperty(float)
    def t(self):
        return self._t

    @t.setter
    def t(self, valor):
        self._t = valor
        self.update()

    def atualizar(self):
        self._t += 0.02
        if self._t >= 2 * np.pi:
            self._t = 0
            self.alternar_transformacao()
        self.cena.update()

    def alternar_transformacao(self):
        if self.transformacao == 'caleidoscopio':
            self.transformacao = 'headset'
        else:
            self.transformacao = 'caleidoscopio'

    def drawForeground(self, painter, rect):
        painter.setRenderHint(QPainter.Antialiasing)
        centro = rect.center()
        raio_maximo = min(rect.width(), rect.height()) / 3
        for forma in self.formas:
            fase = np.sin(self._t) * np.pi
            if self.transformacao == 'caleidoscopio':
                raio = raio_maximo * np.sin(self._t + fase)
                angulo = np.radians(random.uniform(0, 360))
                x = centro.x() + raio * np.cos(angulo)
                y = centro.y() + raio * np.sin(angulo)
            else:  # Transformação para 'headset'
                x = centro.x() + np.cos(self._t + fase) * 100
                y = centro.y() + np.sin(self._t + fase) * 50
            tamanho = forma['tamanho'] * (0.5 + 0.5 * np.sin(self._t + fase))
            painter.setBrush(QBrush(forma['cor']))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(x, y), tamanho, tamanho)

class JanelaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector de Notas Musicais")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.visao_caleidoscopio = VisaoCaleidoscopio()
        self.layout.addWidget(self.visao_caleidoscopio)

        self.label = QLabel("Pressione 'Iniciar' para capturar áudio")
        self.label.setFont(QFont('Arial', 16))
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.botao_iniciar = QPushButton("Iniciar")
        self.botao_iniciar.setFont(QFont('Arial', 14))
        self.botao_iniciar.clicked.connect(self.iniciar_captura)
        self.layout.addWidget(self.botao_iniciar)

        self.botao_parar = QPushButton("Parar")
        self.botao_parar.setFont(QFont('Arial', 14))
        self.botao_parar.clicked.connect(self.parar_captura)
        self.layout.addWidget(self.botao_parar)
        self.botao_parar.setDisabled(True)

        self.grafico_partitura = QGraphicsView()
        self.cena_partitura = QGraphicsScene(self)
        self.grafico_partitura.setScene(self.cena_partitura)
        self.layout.addWidget(self.grafico_partitura)

        self.label_desenvolvido = QLabel("Desenvolvido por Moésio Fiùza")
        fonte_negrito = QFont('Arial', 10)
        fonte_negrito.setBold(True)
        self.label_desenvolvido.setFont(fonte_negrito)
        self.label_desenvolvido.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.label_desenvolvido)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        self.processador_audio = ProcessadorAudio()
        self.processador_audio.nota_detectada.connect(self.atualizar_partitura)
        self.processador_audio.captura_finalizada.connect(self.gerar_partitura_pdf)

    def iniciar_captura(self):
        self.label.setText("Capturando áudio...")
        self.processador_audio.start()
        self.visao_caleidoscopio.iniciar_animacao()
        self.botao_iniciar.setDisabled(True)
        self.botao_parar.setDisabled(False)

    def parar_captura(self):
        self.label.setText("Processando notas capturadas...")
        self.processador_audio.parar()
        self.botao_iniciar.setDisabled(False)
        self.botao_parar.setDisabled(True)

    def atualizar_partitura(self, nota):
        self.label.setText(f"Nota detectada: {nota.nameWithOctave}")
        self.gerar_imagem_partitura()

    def gerar_imagem_partitura(self):
        partitura = stream.Stream(self.processador_audio.partitura_musical)
        partitura.write('lilypond', fp='partitura.ly')
        try:
            subprocess.run([environment.UserSettings()['lilypondPath'], 'partitura.ly'], check=True)
            pixmap = QPixmap("partitura.pdf")
            if not pixmap.isNull():
                self.cena_partitura.clear()
                self.cena_partitura.addPixmap(pixmap)
                self.grafico_partitura.fitInView(self.cena_partitura.itemsBoundingRect(), Qt.KeepAspectRatio)
            else:
                print("Erro ao carregar a imagem da partitura.")
        except subprocess.CalledProcessError as e:
            self.label.setText(f"Erro ao gerar a partitura: {str(e)}")

    def gerar_partitura_pdf(self):
        caminho, _ = QFileDialog.getSaveFileName(self, "Salvar Partitura", "", "PDF Files (*.pdf)")
        if caminho:
            ly_file = caminho.replace('.pdf', '') + '.ly'
            self.processador_audio.partitura_musical.write('lilypond', fp=ly_file)
            try:
                subprocess.run([environment.UserSettings()['lilypondPath'], ly_file], check=True)
                self.label.setText(f"Partitura salva em: {caminho.replace('.ly', '.pdf')}")
            except subprocess.CalledProcessError as e:
                self.label.setText(f"Erro ao salvar a partitura: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_teal.xml')

    janela = JanelaPrincipal()
    janela.show()
    sys.exit(app.exec_())
