import subprocess
import sys
import time
import os
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DOS CAMINHOS ---
# Define a pasta onde os scripts de treinamento estão localizados.
scripts_folder = "efficientnet"

# Lista com os nomes dos scripts para executar em ordem.
script_filenames = [
    "efficientnet_finetuningbalanced.py",
    "efficientnet_finetuning_data-aug_balanced.py"
]

# Monta o caminho completo para cada script
scripts_to_run = [os.path.join(scripts_folder, filename) for filename in script_filenames]


print("="*60)
print("INICIANDO A EXECUÇÃO DE TODOS OS TREINAMENTOS")
print(f"Executando a partir do diretório: {os.getcwd()}")
print(f"Serão executados {len(scripts_to_run)} scripts da pasta '{scripts_folder}'.")
print("Haverá uma pausa de 15 minutos entre cada execução.")
print("="*60)
print("\n")

start_time_total = time.time()

# Itera sobre a lista e executa cada script
for i, script_path in enumerate(scripts_to_run):
    print(f"--- [{(i+1)}/{len(scripts_to_run)}] EXECUTANDO: {script_path} ---")
    start_time_script = time.time()

    try:
        # Usa 'sys.executable' para garantir que está usando o mesmo interpretador Python
        # (importante para ambientes virtuais como o seu .venv)
        subprocess.run([sys.executable, script_path], check=True, text=True, capture_output=False)
        
        end_time_script = time.time()
        duration_script = end_time_script - start_time_script
        print(f"\n--- SUCESSO! '{script_path}' finalizado em {duration_script:.2f} segundos. ---")

        # --- LÓGICA DA PAUSA ---
        # Verifica se este NÃO é o último script da lista para iniciar a pausa.
        if i < len(scripts_to_run) - 1:
            pause_minutes = 15
            pause_seconds = pause_minutes * 60
            
            next_start_time = datetime.now() + timedelta(seconds=pause_seconds)
            
            print(f"\n--- PAUSANDO POR {pause_minutes} MINUTOS ---")
            print(f"O próximo script ('{scripts_to_run[i+1]}') começará por volta das {next_start_time.strftime('%H:%M:%S')}.")
            
            # Pausa a execução pelo tempo definido
            time.sleep(pause_seconds)
            
            print("\n--- Pausa finalizada. Retomando a execução! ---")
        
        print("\n" + "="*60 + "\n")

    except FileNotFoundError:
        print(f"\n*** ERRO: O arquivo '{script_path}' não foi encontrado. ***")
        print("Verifique se o nome e o caminho estão corretos.")
        break # Para a execução dos demais scripts
    except subprocess.CalledProcessError as e:
        print(f"\n*** ERRO: O script '{script_path}' falhou durante a execução. ***")
        print("Revise o output acima para identificar a causa do erro.")
        break # Para a execução dos demais scripts
    except Exception as e:
        print(f"\n*** Ocorreu um erro inesperado ao executar '{script_path}': {e} ***")
        break

end_time_total = time.time()
duration_total = end_time_total - start_time_total

print("\n" + "="*60)
print("TODOS OS TREINAMENTOS FORAM CONCLUÍDOS!")
print(f"Tempo total de execução (incluindo pausas): {(duration_total/60):.2f} minutos.")
print("="*60)