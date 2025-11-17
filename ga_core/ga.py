# ga.py

import random
import time
import asyncio
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from agents.llm_agent import LLMAgent
from agents.generate_data import generar_data_con_ollama
from agents.regenerate_prompt import obtener_prompt_regenerado
from metrics.bert import bertscore_individuos
from operadores.crossover import crossover
from operadores.mutation import mutacion
from metrics.reports import append_metrics

# === CONFIGURACIÓN ===
CHILD_BATCH_SIZE = 10

async def generar_data_para_individuo(individuo, ref_text, llm_agent: 'LLMAgent', temperatura=0.7):
    out = await generar_data_con_ollama(
        individuo, texto_referencia=ref_text, llm_agent=llm_agent, temperatura=temperatura
    )
    individuo["generated_data"] = out.data.strip() if out else ""
    return individuo

async def regenerar_prompt(individuo, ref_text, llm_agent: 'LLMAgent', temperatura=0.9):
    if individuo.get("keywords"):
        prompts = await obtener_prompt_regenerado(
            texto_referencia=ref_text, role=individuo["role"], topic=individuo["topic"],
            keywords=individuo["keywords"], n=1, llm_agent=llm_agent, temperatura=temperatura
        )
        individuo["prompt"] = prompts[0] if prompts else ""
    else:
        individuo["prompt"] = ""
    return individuo

def torneo(individuos, k=3):
    candidatos = random.sample(individuos, k)
    return max(candidatos, key=lambda x: x["fitness"])

# Función asíncrona para procesar un solo hijo
async def procesar_hijo(hijo: dict, ref_text: str, llm_agent: 'LLMAgent', prob_mutacion: float):
    """
    Encapsula toda la lógica de procesamiento (mutación, prompt, data) para un individuo.
    """
    try:
        if random.random() < prob_mutacion:
            hijo = await mutacion(hijo, llm_agent)
        
        hijo = await regenerar_prompt(hijo, ref_text, llm_agent)
        hijo = await generar_data_para_individuo(hijo, ref_text, llm_agent)
        return hijo
    except Exception as e:
        # print(f"⚠️ Error procesando hijo: {e}")
        return None # Retornamos None para que el bucle sepa que falló

async def metaheuristica(individuos, ref_text, llm_agent: 'LLMAgent', bert_model: str,
                   generaciones=5, k=3, prob_crossover=0.8, prob_mutacion=0.1, num_elitismo=2,
                   outdir: Path = Path(".")):
    poblacion = individuos[:]
    poblacion_size = len(poblacion)
    evo_t0 = time.perf_counter()

    for g in range(generaciones):
        gen_t0 = time.perf_counter()
        print(f"🌀 Generación {g+1}/{generaciones}")

        # Elitismo
        poblacion.sort(key=lambda x: x["fitness"], reverse=True)
        nueva_poblacion = poblacion[:num_elitismo]

        # Barra de progreso para hijos nuevos
        hijos_por_evaluar = []
        hijos_necesarios = poblacion_size - len(nueva_poblacion) 
        pbar = tqdm(total=hijos_necesarios, desc=f"Procesando Gen {g+1}", unit="hijos")

        # Llenar hasta completar la población
        while len(hijos_por_evaluar) < hijos_necesarios:
            
            # ¿Cuántos faltan?
            faltantes = hijos_necesarios - len(hijos_por_evaluar)
            # Lote actual (máximo 10)
            batch_size = min(CHILD_BATCH_SIZE, faltantes)
            
            # Crear lote de 'Hijos a Procesar'
            tasks = []
            for _ in range(batch_size):
                if random.random() < prob_crossover:
                    p1, p2 = torneo(poblacion, k=k), torneo(poblacion, k=k)
                    hijo = crossover(p1, p2)
                else:
                    hijo = torneo(poblacion, k=k).copy()
                tasks.append(procesar_hijo(hijo, ref_text, llm_agent, prob_mutacion))
            
            # Ejecutar lote
            resultados = await asyncio.gather(*tasks)
            
            # Filtrar válidos
            validos = [h for h in resultados if h is not None]
            
            # Añadir a la lista de hijos por evaluar
            hijos_por_evaluar.extend(validos)
            pbar.update(len(validos))
        
        pbar.close()
        
        # Evaluamos a TODOS los nuevos individuos en un solo lote
        if hijos_por_evaluar:
            hijos_evaluados = bertscore_individuos(hijos_por_evaluar, ref_text, model_type=bert_model)
            # Añadimos los individuos ya evaluados a la nueva población.
            nueva_poblacion.extend(hijos_evaluados)

        poblacion = nueva_poblacion
        best = max(poblacion, key=lambda x: x["fitness"])
        gen_dt = time.perf_counter() - gen_t0
        print(f"   → Mejor fitness: {best['fitness']:.4f} (Duración: {gen_dt:.2f}s)")
        append_metrics(outdir, g+1, poblacion, duration_sec=gen_dt)
        
    evo_dt = time.perf_counter() - evo_t0
    with open(outdir / "runtime.tmp", "a", encoding="utf-8") as f:
        f.write(f"evolution_sec={evo_dt:.6f}\n")
    return poblacion