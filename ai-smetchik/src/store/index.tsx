import React, { createContext, useCallback, useContext, useMemo, useState } from 'react';

export type RequestStatus = 'new' | 'pending' | 'approved' | 'anomaly';

export type WorkLine = {
  id: string;
  name: string;
  unit: string;
  qty: number;
  price: number;
  rate: string;
};

export type MaterialLine = {
  id: string;
  name: string;
  qty: number;
  unit: string;
  price: number;
};

export type AppRequest = {
  id: number;
  title: string;
  address: string;
  photo: number;
  status: RequestStatus;
  total: number;
  accuracy?: number;
  anomalyNote?: string;
  createdAt: string;
};

export type Draft = {
  photoUri: string | number | null;
  description: string;
  house: string;
};

const HOUSES = ['ЖК Северный, Корпус 1', 'ЖК Северный, Корпус 2', 'ЖК Северный, Корпус 3'];

const WORKS: WorkLine[] = [
  { id: 'w1', name: 'Замена керамической плитки', unit: 'м²', qty: 0.5, price: 450, rate: 'ТЕР 11-01-004-01' },
  { id: 'w2', name: 'Очистка основания', unit: 'м²', qty: 0.5, price: 120, rate: 'ТЕР 11-01-006-02' },
  { id: 'w3', name: 'Облицовка керамогранитом', unit: 'м²', qty: 0.5, price: 1500, rate: 'ТЕР 11-01-007-01' },
];

const MATERIALS: MaterialLine[] = [
  { id: 'm1', name: 'Плитка керамогранит', qty: 1, unit: 'шт', price: 620 },
  { id: 'm2', name: 'Клей плиточный', qty: 2, unit: 'кг', price: 95 },
  { id: 'm3', name: 'Затирка для швов', qty: 0.3, unit: 'кг', price: 280 },
];

const SEED_REQUESTS: AppRequest[] = [
  {
    id: 1041,
    title: 'Трещина на фасаде',
    address: 'ЖК Северный, Корпус 2',
    photo: require('../../assets/facade-crack.png'),
    status: 'pending',
    total: 3450,
    accuracy: 94,
    createdAt: 'Сегодня, 09:15',
  },
  {
    id: 1040,
    title: 'Течь в паркинге, B2',
    address: 'ЖК Северный, Корпус 2',
    photo: require('../../assets/parking-leak.png'),
    status: 'anomaly',
    total: 12870,
    accuracy: 71,
    anomalyNote: 'Расхождение с ТЕР 18%',
    createdAt: 'Сегодня, 08:42',
  },
  {
    id: 1039,
    title: 'Не работает доводчик',
    address: 'ЖК Северный, Корпус 1',
    photo: require('../../assets/door-closer.png'),
    status: 'approved',
    total: 2150,
    accuracy: 97,
    createdAt: 'Вчера, 17:30',
  },
  {
    id: 1038,
    title: 'Скол плитки в МОП',
    address: 'ЖК Северный, Корпус 3',
    photo: require('../../assets/tile-chip.png'),
    status: 'new',
    total: 1280,
    accuracy: 88,
    createdAt: 'Вчера, 11:05',
  },
];

type Store = {
  requests: AppRequest[];
  draft: Draft;
  works: WorkLine[];
  materials: MaterialLine[];
  houses: string[];
  setDraft: (patch: Partial<Draft>) => void;
  updateWorkQty: (id: string, qty: number) => void;
  approveDraft: (title: string, total: number) => void;
  worksTotal: number;
  materialsTotal: number;
  total: number;
};

const Ctx = createContext<Store>(null as unknown as Store);

export function StoreProvider({ children }: { children: React.ReactNode }) {
  const [requests, setRequests] = useState<AppRequest[]>(SEED_REQUESTS);
  const [draft, setDraftState] = useState<Draft>({ photoUri: null, description: '', house: HOUSES[1] });
  const [works, setWorks] = useState<WorkLine[]>(WORKS);

  const setDraft = useCallback((patch: Partial<Draft>) => {
    setDraftState((s) => ({ ...s, ...patch }));
  }, []);

  const updateWorkQty = useCallback((id: string, qty: number) => {
    setWorks((ws) => ws.map((w) => (w.id === id ? { ...w, qty: Math.max(0, qty) } : w)));
  }, []);

  const approveDraft = useCallback(
    (title: string, total: number) => {
      setRequests((rs) => {
        const maxId = Math.max(...rs.map((r) => r.id));
        return [
          {
            id: maxId + 1,
            title,
            address: draft.house,
            photo: require('../../assets/tile-chip.png') as number,
            status: 'approved' as RequestStatus,
            total,
            accuracy: 92,
            createdAt: 'Только что',
          },
          ...rs,
        ];
      });
    },
    [draft.house]
  );

  const worksTotal = useMemo(() => works.reduce((s, w) => s + w.qty * w.price, 0), [works]);
  const materialsTotal = useMemo(() => MATERIALS.reduce((s, m) => s + m.qty * m.price, 0), []);
  const total = worksTotal + materialsTotal;

  const value = useMemo(
    () => ({ requests, draft, works, materials: MATERIALS, houses: HOUSES, setDraft, updateWorkQty, approveDraft, worksTotal, materialsTotal, total }),
    [requests, draft, works, setDraft, updateWorkQty, approveDraft, worksTotal, materialsTotal, total]
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useStore() {
  return useContext(Ctx);
}

export function formatMoney(n: number) {
  return Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
}

export function imgSource(v: string | number) {
  return typeof v === 'string' ? { uri: v } : v;
}
