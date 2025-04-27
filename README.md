# Comportamiento del Filtro de Kalman bajo diferentes configuraciones de ruido

Este proyecto analiza el comportamiento del filtro de Kalman en tres configuraciones distintas de ruido:

## Configuraciones

### 1. Ruido Bajo (Configuración predeterminada)

**Parámetros:**

- `proc_noise_std = [0.02, 0.02, 0.01]`
- `obs_noise_std = [0.02, 0.02, 0.01]`

**Descripción:**

- El modelo de movimiento y las mediciones son muy confiables.
- El filtro combina predicción y corrección de forma precisa y estable.
- El estado sigue de cerca el movimiento real y la incertidumbre disminuye rápidamente.

---

### 2. Ruido Alto en la Medición (Q grande)

**Parámetros:**

- `obs_noise_std = [0.5, 0.5, 0.3]`

**Descripción:**

- El filtro confía más en el modelo de movimiento que en las mediciones.
- Las observaciones son consideradas "sospechosas" y tienen menor peso en la corrección.
- El filtro reacciona lentamente a cambios detectados solo por los sensores.
- Ideal cuando las mediciones son muy ruidosas o inestables.

---

### 3. Ruido Alto en el Proceso (R grande)

**Parámetros:**

- `proc_noise_std = [0.5, 0.5, 0.3]`

**Descripción:**

- El modelo de movimiento es considerado poco confiable.
- El filtro depende principalmente de las observaciones para corregir el estado.
- Se ajusta rápidamente a los cambios detectados por los sensores.
- Puede volverse inestable si las observaciones son ruidosas.
