from src.services.weather_service import WeatherService

service = WeatherService()

service.download_weather(

    start_date="2022-01-01",

    end_date="2022-12-31"

)