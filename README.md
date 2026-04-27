# Quadruped Optimal Control: PMP, LQG & MPC

## Integrantes
Emiliano Niño García | A00228130  
Oscar de la Rosa López | A00838666  
Rigoberto Said Soto Quiroga | A01571662  
Arturo Balboa Alvarado | A01712275  
Angel Hernández Rojas | A00836889

# Metodología

## Base de simulación
Este proyecto se desarrolló utilizando el repositorio **Quadruped-PyMPC** como framework base de simulación y control para robots cuadrúpedos en MuJoCo.

Se utilizó específicamente el robot:

- **AlienGo** (`--robot-name aliengo`)

sobre el entorno:

- Flat terrain en MuJoCo
- Simulación dinámica con `sim_dt = 0.002 s`
- Frecuencia de control ≈ 100 Hz

---

## Integración de controladores
Se integraron tres controladores proporcionados originalmente en el curso:

- **PMP** (Pontryagin Minimum Principle)
- **LQG** (Linear Quadratic Gaussian)
- **MPC** (Model Predictive Control)

Estos controladores fueron conectados a la librería `Quadruped-PyMPC`, utilizando la dinámica del robot y el wrapper:

```python
QuadrupedPyMPC_Wrapper(...)
```

permitiendo traducir las fuerzas calculadas por los controladores en torques articulares aplicados al robot AlienGo.

---

## Cambios realizados sobre el ejemplo original del profesor

### 1. De regulación a tracking por waypoints
El ejemplo original estaba enfocado en estabilización.

Nosotros añadimos:

- generación de trayectorias por waypoints:
  - line
  - square
  - zigzag

```bash
--path line
--path square
--path zigzag
```

usando:

```python
WaypointTrajectory(...)
waypoint_follower(...)
```

para que el robot no solo se estabilizara sino siguiera trayectorias.

---

### 2. Integración con locomoción real del PyMPC
En lugar de mover únicamente el centro de masa, se acopló el alto nivel (PMP/LQG/MPC) con el generador de marcha de Quadruped-PyMPC:

- gait trot
- footstep planning
- swing trajectories
- ground reaction forces
- torque computation

Esto permitió locomoción real del cuadrúpedo.

---

### 3. Modificación de parámetros del gait
Se ajustaron parámetros del robot:

- `step_freq`
- `duty_factor`
- `step_height`
- impedance gains
- swing feedback gains

para mejorar estabilidad antes de aumentar velocidad.

---

### 4. Corrección por fuerzas de reacción (GRF feedback)
Se agregó retroalimentación usando:

```python
controller_velocity_correction(...)
```

para corregir:

- velocidad longitudinal
- velocidad lateral
- yaw rate

a partir de las fuerzas de reacción calculadas.

---

### 5. Comparación automática entre controladores
Se añadió modo comparación:

```bash
--controller all
```

que ejecuta:

- PMP
- LQG
- MPC

y genera:

- plots comparativos
- métricas automáticas
- CSV de resultados

```bash
results/metrics_runs.csv
results/comparison_*.csv
```

---

## Métricas evaluadas
Se evaluó desempeño con:

- Tracking RMSE
- Error al waypoint final
- Porcentaje de trayectoria completada
- Survival time
- Error de velocidad
- Distancia recorrida
- Norma de fuerzas GRF

Esto permitió comparar cuantitativamente los tres controladores.

---

## Perturbaciones
También se probaron perturbaciones externas:

```bash
--disturbance impulse
--disturbance persistent
```

para analizar robustez ante empujes y disturbios sostenidos.

---

## Comandos de ejecución

Ejemplo LQG:

```bash
python examples/run_mujoco.py \
--controller lqg \
--robot-name aliengo \
--path line \
--duration 10
```

Comparación completa + disturbance:

```bash
python examples/run_mujoco.py \
--controller all \
--robot-name aliengo \
--path square \
--disturbance impulse
```
Extras

si no se especifica duración la simulación durara hasta que el robot termine todo el recorrido. 
Si MujoCo esta dando problemas es mejor evitar el render y que simule todo.

```bash
--no-render
```
## Resultados
Las graficas que se generan al correr el codigo se guardan en el folder de results, se pueden hacer pruebas individuales o poner el comando de **--controller all** para tener una comparativa de la efectividad de los 3 controladores, para la discución de resultados y determinar efectividad se tomara en cuenta si logro terminar la ruta preestablecida, su error de raiz cuadratico (RMSE), Final waypoint error (FinalWP), Mean velocity error y Mean GRF norm. 

Además de la graficas, al correr nuestro run_mujoco.py genera un summary final de los datos mencionados anteriormente.

### Primera Prueba: Rutas sin perturbaciones
![Line no disturbance](images/mujoco_comparison_aliengo_line_none.png)

