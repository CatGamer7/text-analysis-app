import { test, expect } from '@playwright/test';

test('TC - 001', async ({ request }) => {
    const response = await request.post('http://127.0.0.1:8000/api/v1/models/rubert-emotion-ru/probe' , {data: {text: ''}});

    const result = await response.status();
    expect(result).toBe(422)
});

test('TC - 101', async ({ request }) => {
    const response = await request.post('http://127.0.0.1:8000/api/v1/models/rubert-emotion-ru/probe' , {data: {text: 'Я так рад'}});

    await expect(response).toBeOK();

    const body = await response.json(); 
    expect(body.primary_label).toBe('joy');
});

test('TC - 102', async ({ request }) => {
    const response = await request.post('http://127.0.0.1:8000/api/v1/models/rubert-emotion-ru/probe' , {data: {text: 'Сука, я сегодня забыл зонт дома и промок как петух. Тупой босс отправил меня домой и сказал, что оплаты не будет сегодня. Урод ебанный.'}});

    await expect(response).toBeOK();

    const body = await response.json();
    const labels = ['neutral', 'joy', 'sadness', 'anger', 'fear', 'surprise'];
    expect(labels).toContain(body.primary_label);
});
test('TC - 002', async ({ request }) => {
    const response = await request.post('http://127.0.0.1:8000/api/v1/models/rubert-emotion-ru/probe' , {data: {text: '        '}});

    expect(response.status()).toBe(422);
});

test('TC - 003', async ({ request }) => {
    const texts = ['1234567890', '!@#$%^&*()_+{}:"<>?[];\',./', 'Гора кот веревка стол арбуз. Нефрит бегемот окно быстрый', '今天我忘了带伞，全身都湿透了'];
    const labels = ['neutral', 'joy', 'sadness', 'anger', 'fear', 'surprise'];

    for (const text of texts) {
        const response = await request.post('http://127.0.0.1:8000/api/v1/models/rubert-emotion-ru/probe' , {data: {text: text}});

        await expect(response).toBeOK();

        const body = await response.json();
        expect(labels).toContain(body.primary_label);
    }
});

test('TC - 304', async ({ request }) => {
    const longText = 'а'.repeat(10001);
    const response = await request.post('http://127.0.0.1:8000/api/v1/models/rubert-emotion-ru/probe' , {data: {text: longText}});

    await expect(response).toBeOK();
});

test('TC - 402', async ({ request }) => {
    const modelsResponse = await request.get('http://127.0.0.1:8000/api/v1/models');
    await expect(modelsResponse).toBeOK();

    const models = (await modelsResponse.json()).items;

    for (const model of models) {
        const response = await request.post('http://127.0.0.1:8000/api/v1/models/' + model.id + '/probe' , {data: {text: 'Сегодня я не взял с собой зонт, я весь промок'}});

        await expect(response).toBeOK();

        const body = await response.json();
        expect(model.labels).toContain(body.primary_label);
    }
});

test('TC - 403', async ({ request }) => {
    const text = 'Сегодня я не взял с собой зонт перед выходом на улицу, я весь промок.';

    const first = await request.post('http://127.0.0.1:8000/api/v1/models/rubert-emotion-ru/probe' , {data: {text: text}});
    const second = await request.post('http://127.0.0.1:8000/api/v1/models/rubert-emotion-ru/probe' , {data: {text: text}});

    const firstBody = await first.json();
    const secondBody = await second.json();
    expect(secondBody.scores).toEqual(firstBody.scores);
});

test('Несуществующая модель', async ({ request }) => {
    const response = await request.post('http://127.0.0.1:8000/api/v1/models/blabla/probe' , {data: {text: 'Я так рад'}});

    expect(response.status()).toBe(404);
});
