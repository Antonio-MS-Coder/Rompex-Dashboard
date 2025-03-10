# Instrucciones para subir el Dashboard a GitHub

Sigue estos pasos para crear un repositorio en GitHub y subir tu código:

## 1. Crear un repositorio en GitHub

1. Ve a [GitHub](https://github.com/) y asegúrate de iniciar sesión.
2. Haz clic en el botón "+" en la esquina superior derecha y selecciona "New repository".
3. Nombra tu repositorio "Rompex-Dashboard".
4. Añade una descripción: "Dashboard para evaluación de ubicación de planta de distribución".
5. Deja el repositorio como público.
6. No inicialices el repositorio con un README, .gitignore o licencia, ya que ya tenemos nuestros archivos localmente.
7. Haz clic en "Create repository".

## 2. Conectar tu repositorio local con GitHub

Una vez creado el repositorio, GitHub te mostrará comandos para conectar tu repositorio local con el remoto. Ejecuta los siguientes comandos en tu terminal, reemplazando `TU_USUARIO` con tu nombre de usuario de GitHub:

```bash
git remote add origin https://github.com/TU_USUARIO/Rompex-Dashboard.git
git push -u origin main
```

Si prefieres usar SSH en lugar de HTTPS (recomendado si tienes configurada una clave SSH):

```bash
git remote add origin git@github.com:TU_USUARIO/Rompex-Dashboard.git
git push -u origin main
```

## 3. Verificar que todo se haya subido correctamente

1. Visita `https://github.com/TU_USUARIO/Rompex-Dashboard` en tu navegador.
2. Deberías ver todos tus archivos en el repositorio.

## 4. Actualizar el repositorio en el futuro

Cada vez que hagas cambios en tu código y quieras actualizarlos en GitHub, ejecuta:

```bash
git add .
git commit -m "Descripción de los cambios"
git push
```

## 5. Compartir tu Dashboard

Ahora puedes compartir tu Dashboard con cualquier persona enviándoles el enlace a tu repositorio de GitHub.

## Nota sobre la ejecución del Dashboard

Para que otras personas puedan ejecutar tu Dashboard, deberán:

1. Clonar tu repositorio: `git clone https://github.com/TU_USUARIO/Rompex-Dashboard.git`
2. Crear un entorno virtual: `python3 -m venv venv`
3. Activar el entorno virtual:
   - En macOS/Linux: `source venv/bin/activate`
   - En Windows: `venv\Scripts\activate`
4. Instalar las dependencias: `pip install -r requirements.txt`
5. Ejecutar la aplicación: `python app.py`
6. Abrir en el navegador: `http://127.0.0.1:8050/` 