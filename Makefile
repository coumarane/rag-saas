.PHONY: dev test build migrate stop clean logs shell

dev:
	docker compose up --build -d --wait
	@echo "Services starting..."
	@echo "API: http://localhost:8000"
	@echo "Frontend: http://localhost:3000"
	@echo "Flower: http://localhost:5555"
	@echo "MinIO Console: http://localhost:9001"

migrate:
	docker compose exec api alembic upgrade head

test:
	docker compose exec api pytest tests/ -v

build:
	docker compose build

stop:
	docker compose down

clean:
	docker compose down -v --remove-orphans

logs:
	docker compose logs -f

shell:
	docker compose exec api bash
