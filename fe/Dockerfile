FROM node:18-alpine

WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/public /app/src

COPY package*.json ./
RUN npm install

# Copy project files
COPY public/ ./public/
COPY src/ ./src/
COPY tsconfig.json .

EXPOSE 3000

CMD ["npm", "start"] 