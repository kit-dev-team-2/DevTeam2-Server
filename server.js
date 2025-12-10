// server3.js
const express = require('express');
const http = require('http');
const WebSocket = require('ws');

const app = express();
app.get('/', (_, res) => res.send('WS server OK'));
const server = http.createServer(app);

const wss = new WebSocket.Server({ server });

const PORT = process.env.PORT || 8080;
const HOST = process.env.HOST || '0.0.0.0';

wss.on('connection', (ws, req) => {
    const ip = req.socket.remoteAddress;
    ws._ip = ip;
    console.log('------------------------------');
    console.log('WS connected:', ip);
    console.log('------------------------------');

    ws.on('message', (data, isBinary) => {
        if (isBinary) {
            console.log('RX bin:', data.length, 'bytes');
            ws.send(JSON.stringify({ type: 'ack-bin', bytes: data.length }));
            return;
        }

        const msg = data.toString();
        let payload = null;
        try {
            payload = JSON.parse(msg);
        } catch (e) {
            console.log('[RAW TEXT]', msg);
            return;
        }

        const type = payload.type;

        // 0) pong → RTT만 로그
        if (type === 'pong') {
            const t = payload.t || 0;
            const rtt = Date.now() - t;
            console.log(`[PING] ${ip} rtt=${rtt}ms`);
            return;
        }

        // 1) detection → 여기에서 분류 결과 로깅
        if (type === 'detection') {
            console.log('[DETECT]', JSON.stringify(payload));

            wss.clients.forEach((client) => {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(JSON.stringify(payload));
                }
            });
            return;
        }

        // 2) config_update → 설정 변경 로그
        if (type === 'config_update') {
            console.log('[CONFIG]', JSON.stringify(payload));

            wss.clients.forEach((client) => {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(JSON.stringify(payload));
                }
            });
            return;
        }

        // 기타
        console.log('[OTHER]', payload);
    });

    ws.on('close', () => console.log('WS closed', ip));

    const iv = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping', t: Date.now() }));
        } else {
            clearInterval(iv);
        }
    }, 3000);
});

server.listen(PORT, HOST, () => {
    console.log(`HTTP/WS on http://${HOST}:${PORT}`);
});
