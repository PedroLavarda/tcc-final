import tensorflow as tf
import ctypes
import os
import sys

print("--- INICIANDO DIAGNÓSTICO DE GPU PARA TENSORFLOW MODERNO (TF 2.12+) ---")
print("\n--- Verificação 1: Versões de Software ---")
print(f"Versão do Python: {sys.version}")
print(f"Versão do TensorFlow: {tf.__version__}")
print("--------------------")

print("\n--- Verificação 2: Detecção de GPU pelo TensorFlow ---")
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[SUCESSO] TensorFlow detectou {len(gpus)} GPU(s): {gpus}")
    else:
        print("[FALHA] tf.config.list_physical_devices('GPU') retornou uma lista vazia.")
        print("Isso significa que o TensorFlow não encontrou uma GPU compatível.")
except Exception as e:
    print(f"[ERRO] Ocorreu uma exceção ao tentar listar as GPUs: {e}")
print("--------------------")

print("\n--- Verificação 3: Carregamento Manual das Bibliotecas (DLLs) para CUDA 12.x ---")
print("Esta etapa tenta carregar as bibliotecas que o TensorFlow 2.12+ precisa.")
print("Se uma DLL não for encontrada, significa que ela não está no PATH do sistema.")

# No Windows, o PATH é usado para encontrar DLLs.
# A biblioteca principal do CUDA 12 é cudart64_12.dll
# A biblioteca principal do cuDNN para CUDA 12 ainda pode se chamar cudnn64_8.dll,
# mas há outras mais específicas que podemos verificar.
dlls_to_check = {
    'cudart64_12.dll': {
        'message': "Biblioteca principal do CUDA 12.x Runtime.",
        'fix': ("Verifique se o CUDA Toolkit 12.2 está instalado e se "
                "'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.2\\bin' "
                "está nas Variáveis de Ambiente (PATH). Reinicie o PC após adicionar.")
    },
    'cudnn64_8.dll': {
        'message': "Biblioteca principal do cuDNN.",
        'fix': ("Verifique se você copiou os arquivos do cuDNN para CUDA 12.x para a pasta do CUDA 12.2. "
                "O arquivo 'cudnn64_8.dll' deve estar em 'C:\\...\\CUDA\\v12.2\\bin'.")
    }
}

found_any = False
for dll, info in dlls_to_check.items():
    try:
        # Tenta carregar a DLL
        ctypes.WinDLL(dll)
        print(f"[SUCESSO] Encontrada e carregada a biblioteca '{dll}'. ({info['message']})")
        found_any = True
    except OSError:
        print(f"[FALHA] Não foi possível encontrar ou carregar a biblioteca '{dll}'.")
        print(f"  - SOLUÇÃO: {info['fix']}")

if not found_any:
     print("\nAVISO: Nenhuma das DLLs principais do CUDA/cuDNN foi encontrada. Este é provavelmente o problema principal.")
else:
     print("\nAVISO: Pelo menos uma DLL foi encontrada. Se a detecção da GPU ainda falhar, o problema pode ser uma incompatibilidade de versão do driver ou do cuDNN.")


print("\n--- DIAGNÓSTICO CONCLUÍDO ---")
